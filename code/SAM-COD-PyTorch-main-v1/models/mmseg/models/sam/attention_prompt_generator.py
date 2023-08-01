import torch
from torch import nn
import numpy as np
import random
from torch.nn import functional as F

from typing import List, Tuple, Type
from visualizer import get_local
from copy import deepcopy

from .common import LayerNorm2d
from typing import Any, Optional, Tuple, List, Generator

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    # 初始偏移，初始点距离图像边缘距离为1 / （2*n_per_side）
    offset = 1 / (2 * n_per_side)
    # 按照网格点点数计算偏移量
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    # 所有点的x轴偏移量
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    # 所有点的y轴偏移量
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    # (32, 32) -> [(32, 32), (32, 32)] -> (32, 32, 2) -> (1024, 1024, 2)
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


class AttentionPromptGenerator(nn.Module):
    """
    Generate mask and points prompt from attention map.
    """
    def __init__(
            self,
            depth: int,
            num_heads: int,
            sample_ratio: int,
            ratio: int,
            scale_factor: int,
            inp_size: tuple,
            embedding_size: tuple,
            method: str,
            mode: str,
            mask_threshold: float,
            num_points: int,
            global_attn_indexes: tuple,
            cat_indexes: list,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth = depth
        self.num_heads = num_heads
        self.ratio = ratio
        self.scale_factor = scale_factor
        self.inp_size = inp_size
        self.embedding_size = embedding_size
        self.method = method
        self.mode = mode
        self.mask_threshold = mask_threshold
        self.num_points = num_points
        self.global_attn_indexes = global_attn_indexes
        self.num_global_attn = len(global_attn_indexes)
        self.num_cat_blocks = self.depth // self.num_global_attn
        self.cat_dim = self.num_cat_blocks * self.num_heads
        self.cat_dim2 = self.cat_dim * self.num_global_attn // ratio
        if mode == '0.0' or mode == '0.1' or mode == '0.2':
            self.blocks = nn.ModuleList()
            for i in range(self.num_global_attn):
                block = Block1(in_dim=self.cat_dim, out_dim=self.cat_dim//ratio, num_heads=self.num_cat_blocks, sample_ratio=sample_ratio)
                self.blocks.append(block)
            self.final_block = Block2(in_dim=self.cat_dim2, out_dim=64, num_heads=self.num_global_attn, sample_ratio=sample_ratio, mode=mode)
            self.pred = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        elif mode == '1':
            self.cat_indexes = cat_indexes
            self.num_cat = len(cat_indexes)
            self.pred1 = nn.Conv2d(self.num_cat * self.num_heads, 1, 1)
        elif mode == '2':
            self.blocks = nn.ModuleList()
            for i in range(self.num_global_attn):
                block = Block1(in_dim=self.cat_dim, out_dim=self.cat_dim//ratio, num_heads=self.num_cat_blocks, sample_ratio=sample_ratio)
                self.blocks.append(block)
            self.final_block = Block1(in_dim=self.cat_dim2, out_dim=64, num_heads=self.num_global_attn, sample_ratio=sample_ratio)
            self.pred = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        elif mode == '3':
            self.cat_indexes = cat_indexes
            self.num_cat = len(cat_indexes)
            self.blocks = nn.Sequential(
                nn.Conv2d(self.num_cat * self.num_heads, self.num_cat * self.num_heads, 3, 1, 1),
                nn.Conv2d(self.num_cat * self.num_heads, 64, 1),
            )
            self.pred = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        elif mode == '4':
            self.cat_indexes = cat_indexes
            self.num_cat = len(cat_indexes)
            self.blocks = Block1(in_dim=self.num_cat * self.num_heads, out_dim=64, num_heads=self.num_cat_blocks, sample_ratio=sample_ratio)
            self.pred = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        elif mode == '5':
            self.blocks1 = nn.ModuleList()
            self.blocks2 = nn.ModuleList()
            self.cat_indexes = cat_indexes
            self.num_cat = len(cat_indexes)
            for i in range(self.num_cat):
                block1 = BasicConv2d(self.num_heads, self.num_heads, 1)
                block2 = BasicConv2d(self.num_heads, self.num_heads, 3, 1, 1)
                self.blocks1.append(block1)
                self.blocks2.append(block2)
            self.final_block = BasicConv2d(self.num_heads, self.num_heads, 3, 1, 1)
            self.pred = nn.Conv2d(self.num_heads, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, attn_list):
        if self.mode == '0.0' or self.mode == '0.1' or self.mode == '0.2':
            attns = []
            for i, blk in enumerate(self.blocks):
                attn_tensor = torch.cat(attn_list[i*self.num_cat_blocks:(i+1)*self.num_cat_blocks], dim=1)
                assert attn_tensor.shape[2:] == self.embedding_size, "Attention size doesn't match image_embedding."
                attn_tensor = blk(attn_tensor)
                attns.append(attn_tensor)
            cat_attns = torch.cat(attns, dim=1)
            cat_attns = self.final_block(cat_attns)
            map = self.pred(cat_attns)
        elif self.mode == '1':
            attns = [attn_list[index] for index in self.cat_indexes]
            cat_attns = torch.cat(attns, dim=1)
            map = self.pred1(cat_attns)
        elif self.mode == '2':
            attns = []
            for i, blk in enumerate(self.blocks):
                attn_tensor = torch.cat(attn_list[i*self.num_cat_blocks:(i+1)*self.num_cat_blocks], dim=1)
                assert attn_tensor.shape[2:] == self.embedding_size, "Attention size doesn't match image_embedding."
                attn_tensor = blk(attn_tensor)
                attns.append(attn_tensor)
            cat_attns = torch.cat(attns, dim=1)
            cat_attns = self.final_block(cat_attns)
            map = self.pred(cat_attns)
        elif self.mode == '3':
            attns = [attn_list[index] for index in self.cat_indexes]
            cat_attns = torch.cat(attns, dim=1)
            cat_attns = self.blocks(cat_attns)
            map = self.pred(cat_attns)
        elif self.mode == '4':
            attns = [attn_list[index] for index in self.cat_indexes]
            cat_attns = torch.cat(attns, dim=1)
            cat_attns = self.blocks(cat_attns)
            map = self.pred(cat_attns)
        elif self.mode == '5':
            for i in range(self.num_cat):
                attn1 = self.blocks1[i](attn_list[self.cat_indexes[i]])
                if i == 0:
                    attn2 = attn1
                    attn2 = attn2 + self.blocks2[i](attn2)
                else:
                    attn2 = attn1 + attn2
                    attn2 = attn2 + self.blocks2[i](attn2)
            attn = self.final_block(attn2)
            map = self.pred(attn)

        pred = torch.sigmoid(map)
        pred = F.interpolate(pred, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        if self.method == 'random':
            binary_mask = pred > self.mask_threshold
            points = self.get_random_points_from_gts(binary_mask, self.num_points)
        elif self.method == 'gumbel':
            points = self.gumbel_softmax_sample(map, self.num_points)

        return pred, points

    def get_random_points_from_gts(self, gts, num):
        batch_size, _, H, W = gts.shape
        gt_array = np.uint8(gts.data.cpu().numpy() * 255)
        batch_centers = []
        batch_labels = []
        for i in range(batch_size):
            gt = gt_array[i, 0, :, :]
            centers = []
            nonzero_coords = np.transpose(np.nonzero(gt))
            if nonzero_coords.shape[0] == 0:
                centers = [[0, 0] for _ in range(num)]
                label = [0 for _ in range(num)]
                batch_centers.append(centers)
                batch_labels.append(label)
                continue
            for _ in range(num):
                selected_pixel = random.choice(nonzero_coords)
                selected_pixel = (selected_pixel[1], selected_pixel[0])
                centers.append([selected_pixel[0], selected_pixel[1]])
            label = [1 for _ in range(num)]
            batch_centers.append(centers)
            batch_labels.append(label)
        batch_centers = np.array(batch_centers)
        batch_labels = np.array(batch_labels)

        points_coords = self.apply_coords(batch_centers, (H, W))
        coords_torch = torch.as_tensor(points_coords, dtype=torch.float, device=self.device)
        labels_torch = torch.as_tensor(batch_labels, dtype=torch.int, device=self.device)

        points = (coords_torch, labels_torch)
        return points

    def gumbel_softmax_sample(self, maps, num):
        B, _, H, W = maps.shape
        maps_reshape = maps.reshape(B, -1, H*W)
        samples = torch.cat([F.gumbel_softmax(logits=maps_reshape, tau=0.5, hard=True, dim=-1).reshape(B, -1, H, W) for _ in range(num)], dim=1)
        index = torch.argwhere(samples == 1)
        batch_points = np.array(index[:, 2:].reshape(B, num, -1).contiguous().cpu())
        batch_labels = torch.zeros((B, num))
        for i in range(index.shape[0]):
            b, c, h, w = index[i, :]
            batch_labels[b, c] = samples[b, c, h, w]
        points_coords = self.apply_coords(batch_points, (H, W))
        coords_torch = torch.as_tensor(points_coords, dtype=torch.float, device=self.device)
        points = (coords_torch, batch_labels)

        return points


    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = get_preprocess_shape(
            original_size[0], original_size[1], self.inp_size
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

class BasicConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
        ))
        self.add_module(name="bn", module=nn.BatchNorm2d(out_channels))
        self.add_module(name="gelu", module=nn.GELU())

class Block1(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, sample_ratio):
        super().__init__()
        self.dim = in_dim
        self.out_dim = out_dim
        self.ca = ChannelAttention(self.dim)
        self.sa = SpatialAttention()
        self.rfb = RFB(in_channel=self.dim, out_channel=self.out_dim)

    def forward(self, x):
        x = x.mul(self.ca(x))
        x = x.mul(self.sa(x))
        x = self.rfb(x)

        return x

class Block2(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, sample_ratio, mode):
        super().__init__()
        self.dim = in_dim
        self.out_dim = out_dim
        self.rfb = RFB(in_channel=self.dim, out_channel=self.out_dim)
        self.norm1 = LayerNorm2d(out_dim)
        if mode == '0.0':
            self.deformable = DeformableAttention(dim=self.out_dim, num_heads=num_heads, sample_ratio=sample_ratio)
        elif mode == '0.1':
            self.deformable = DeformableAttention2(dim=self.out_dim, num_heads=num_heads, sample_ratio=sample_ratio)
        elif mode == '0.2':
            self.deformable = Attention(dim=self.out_dim, num_heads=num_heads)
        self.norm2 = LayerNorm2d(out_dim)
        self.light_mlp = nn.Sequential(
            nn.Conv2d(self.out_dim, 2*self.out_dim, 1),
            nn.Conv2d(2 * self.out_dim, self.out_dim, 1),
        )

    def forward(self, x):
        x = self.rfb(x)
        shortcut = x
        x = self.deformable(self.norm1(x))
        x = x + shortcut
        x = x + self.light_mlp(self.norm2(x))

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道均值和最大值，得到均值mask和最大值mask，拼接后卷积、sigmoid得到空间注意力mask
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RFB(nn.Module):
    """ receptive field block """

    def __init__(self, in_channel, out_channel=64):
        super(RFB, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)  # 当kernel=3，如果dilation=padding则shape不变
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        out = self.gelu(x_cat + self.conv_res(x))
        return out

class Attention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        B, C, H, W = x.shape

        q = self.query(x).reshape(B, self.num_heads, C // self.num_heads, H*W)
        q = q.reshape(B*self.num_heads, -1, H*W).permute(0, 2, 1).contiguous()
        k = self.key(x).reshape(B, self.num_heads, C//self.num_heads, H*W)
        k = k.reshape(B*self.num_heads, -1, H*W)
        v = self.value(x)
        v = v.reshape(B, self.num_heads, C//self.num_heads, H*W)
        v = v.reshape(B*self.num_heads, -1, H*W).permute(0, 2, 1).contiguous()

        attn = (q * self.scale) @ k
        attn = attn.softmax(dim=-1)

        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        x = self.proj(x)

        return x

class DeformableAttention(nn.Module):

    def __init__(self, dim, num_heads, sample_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sample_ratio = sample_ratio
        self.sample_q = nn.Conv2d(dim, sample_ratio, 1)
        self.sample_k = nn.Conv2d(dim, sample_ratio, 1)
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        sample_q = F.gumbel_softmax(self.sample_q(x), tau=1, hard=False, dim=1)[:, [0], :, :]
        sample_k = F.gumbel_softmax(self.sample_k(x), tau=1, hard=False, dim=1)[:, [0], :, :]

        q = (sample_q * self.query(x)).reshape(B, self.num_heads, C // self.num_heads, H*W)
        q = q.reshape(B*self.num_heads, -1, H*W).permute(0, 2, 1).contiguous()
        k = (sample_k * self.key(x)).reshape(B, self.num_heads, C//self.num_heads, H*W)
        k = k.reshape(B*self.num_heads, -1, H*W)
        v = self.value(x)
        shortcut = v
        v = v.reshape(B, self.num_heads, C//self.num_heads, H*W)
        v = v.reshape(B*self.num_heads, -1, H*W).permute(0, 2, 1).contiguous()

        attn = (q * self.scale) @ k
        attn = attn.softmax(dim=-1)

        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        x = self.proj(sample_q * x + (1 - sample_q) * shortcut)

        return x

class DeformableAttention2(nn.Module):

    def __init__(self, dim, num_heads, sample_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sample_ratio = sample_ratio
        self.sample_q = nn.Conv2d(dim, sample_ratio, 1)
        self.sample_k = nn.Conv2d(dim, sample_ratio, 1)
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        sample_q = F.gumbel_softmax(self.sample_q(x), tau=1, hard=True, dim=1)[:, [0], :, :]
        sample_k = F.gumbel_softmax(self.sample_k(x), tau=1, hard=True, dim=1)[:, [0], :, :]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        for i in range(B):
            indices1 = torch.nonzero(sample_q[i])
            indices2 = torch.nonzero(sample_k[i])
            filtered_tensor1 = q[i, :, indices1[:, 1], indices1[:, 2]].reshape(self.num_heads, C//self.num_heads, -1).permute(0, 2, 1).contiguous()
            filtered_tensor2 = k[i, :, indices2[:, 1], indices2[:, 2]].reshape(self.num_heads, C//self.num_heads, -1)
            filtered_tensor3 = v[i, :, indices2[:, 1], indices2[:, 2]].reshape(self.num_heads, C//self.num_heads, -1).permute(0, 2, 1).contiguous()
            attn = (filtered_tensor1 * self.scale) @ filtered_tensor2
            attn = attn.softmax(dim=-1)
            result = (attn @ filtered_tensor3).permute(0, 2, 1).view(C, -1)
            v[i, :, indices1[:, 1], indices1[:, 2]] = result
        x = self.proj(v)

        return x

class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Gumbel(nn.Module):
    '''
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        #logger.add('gumbel_noise', gumbel_noise)
        #logger.add('gumbel_temp', gumbel_temp)

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft
        assert not torch.any(torch.isnan(hard))
        return hard

