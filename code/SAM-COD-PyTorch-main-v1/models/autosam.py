import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visualizer import get_local
get_local.activate()

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer, ContrastivePromptGenerator, PromptEncoder


logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple, List, Generator
from copy import deepcopy


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

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

def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...], target_length: int) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], target_length
    )
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam_auto')
class AutomaticSAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.point_per_side = 32
        self.points_per_batch = 64
        self.inp_size = inp_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.prompt_encoder = PromptEncoder(
            embed_dim=encoder_mode['prompt_embed_dim'],
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=inp_size,
            mask_in_chans=16,
            activation=nn.GELU,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False



        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs=1
        orig_size = self.input.shape[2:]

        """
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        # 密集提示输入 利用特殊嵌入表示没有密集提示
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        """

        # image_encoder ImageEncoderViT
        self.features = self.image_encoder(self.input)

        # Embed prompts
        # 稀疏提示嵌入 构建合适维度的张量，维度为(1, 0, dim) 说明点的组数为1，每组的点数为0，既没有提示点
        points_scale = np.array(orig_size)
        points_for_image = build_all_layer_point_grids(self.point_per_side, 0, 1)[0] * points_scale

        batch_size = self.features.shape[0]
        for i in range(batch_size):

            batch_masks = []
            feature = self.features[i].unsqueeze(0)
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                transformed_points = apply_coords(points, orig_size, self.inp_size)
                in_points = torch.as_tensor(transformed_points, device=self.device)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=self.device)
                in_points = in_points[:, None, :]
                in_labels = in_labels[:, None]
                points = (in_points, in_labels)

                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )

                # Predict masks
                # 图像嵌入  位置编码  稀疏提示嵌入  密集提示嵌入  非多mask输出
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=feature,
                    image_pe=self.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Upscale the masks to the original image resolution
                # masks双线性插值至输入图像大小，根据原图像大小裁剪masks，去掉padding区域
                masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
                batch_masks.append(masks)

            batch_masks = torch.stack(batch_masks, dim=1)
            self.pred_mask = batch_masks

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # 计算BCE Loss
        # self.pred_mask是预测mask
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)

        # 反向传播
        self.loss_G.backward()

    # 前向传播 损失计算 反向传播 权重更新
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad