import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

# 多卡设置
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '8888'vi
# 单机单卡设置
#os.environ["RANK"] = "0"
#os.environ['WORLD_SIZE'] = '4'

# 多卡并行初始化，Linux系统可用mpi,nccl，windows系统用gloo
#torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
# 返回当前进程在提供的组或默认组中的排名，从0到world_size
local_rank = int(os.environ['LOCAL_RANK'])
#local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=12, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


# 验证指标
def eval_psnr(loader, model, eval_type=None):
    with torch.no_grad():
        model.eval()

        if eval_type == 'f1':
            metric_fn = utils.calc_f1
            metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
        elif eval_type == 'fmeasure':
            metric_fn = utils.calc_fmeasure
            metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
        elif eval_type == 'ber':
            metric_fn = utils.calc_ber
            metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
        elif eval_type == 'cod':
            metric_fn = utils.calc_cod
            # S-measure、E-measure、wighted F-measure、MAE
            metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

        if local_rank == 0:
            # pbar = tqdm(total=len(loader), leave=False, desc='val')
            pbar = None
        else:
            pbar = None

        #pred_list = []
        #gt_list = []
        batch_result1, batch_result2, batch_result3, batch_result4 = 0, 0, 0, 0
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = batch['inp']
            gt = batch['gt']

            pred = torch.sigmoid(model.infer(inp))

            result_tmp1, result_tmp2, result_tmp3, result_tmp4 = metric_fn(pred, gt)
            B = pred.shape[0]
            batch_result1 += B * result_tmp1
            batch_result2 += B * result_tmp2
            batch_result3 += B * result_tmp3
            batch_result4 += B * result_tmp4

            #batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            #batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

            #dist.all_gather(batch_pred, pred)
            #pred_list.extend(batch_pred)
            #dist.all_gather(batch_gt, batch['gt'])
            #gt_list.extend(batch_gt)
            #gather_pred = torch.cat(batch_pred, 0)
            #gather_gt = torch.cat(batch_gt, 0)
            #result_tmp1, result_tmp2, result_tmp3, result_tmp4 = metric_fn(gather_pred, gather_gt)
            #result1 += gather_pred.shape[0] * result_tmp1
            #result2 += gather_pred.shape[0] * result_tmp2
            #result3 += gather_pred.shape[0] * result_tmp3
            #result4 += gather_pred.shape[0] * result_tmp4
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        gather_results1 = [0 for _ in range(dist.get_world_size())]
        gather_results2 = [0 for _ in range(dist.get_world_size())]
        gather_results3 = [0 for _ in range(dist.get_world_size())]
        gather_results4 = [0 for _ in range(dist.get_world_size())]
        dist.all_gather_object(gather_results1, batch_result1)
        dist.all_gather_object(gather_results2, batch_result2)
        dist.all_gather_object(gather_results3, batch_result3)
        dist.all_gather_object(gather_results4, batch_result4)
        #pred_list = torch.cat(pred_list, 0)
        #gt_list = torch.cat(gt_list, 0)
        #result1, result2, result3, result4 = metric_fn(pred_list, gt_list)
        result1, result2, result3, result4 = sum(gather_results1) / len(loader.dataset), sum(gather_results2) / len(loader.dataset), sum(gather_results3) / len(loader.dataset), sum(gather_results4) / len(loader.dataset)
    torch.cuda.empty_cache()

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    # 从训练断点开始训练
    if config.get('resume') is not None:
        # model -- sam
        # 载入模型超参
        model = models.make(config['model']).cuda()
        # 设置优化器
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        # 重头开始训练
        # 载入整个模型
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    # 训练epoch数
    max_epoch = config.get('epoch_max')
    # 学习率调整  余弦
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    # 显示进度条
    if local_rank == 0:
        # pbar = tqdm(total=len(train_loader), leave=False, desc='train')
        pbar = None
    else:
        pbar = None

    loss_list = []
    # 输入batch 类型dict
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        # 输入图像
        inp = batch['inp']
        # ground truth
        gt = batch['gt']
        # input和mask传递到GPU
        model.set_input(inp, gt)
        # 前向传播 反向传播 梯度计算 权重更新
        model.optimize_parameters()
        # 数据分布损失
        # batch_loss每个元素维度与model.loss_G每个rank的维度相同
        # all_gather不会进行反向传播
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        # 将python对象序列化为yaml
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # 模型、优化器、起始世代、学习率 初始化
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    """
    打印模型层
    for name, module in model._modules.items():
        print(name, ':', module)
    
    for name in model.named_modules():
        print(name)
        
    for name in (model.children()):
        print(name)
        
    for index, i in model.state_dict().items():
        print(index, i.shape)
        
    删除模型层
    modules = list(model.children())[:-4]
    model = torch.nn.Sequential(*modules)
    
    del model.prompt_generator
    """

    model = model.cuda()
    # 数据并行
    """
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    """
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    # 预训练参数
    sam_checkpoint = torch.load(config['sam_checkpoint'], map_location='cpu')
    model.load_state_dict(sam_checkpoint, strict=False)

    # 设置image_encoder不进行梯度更新
    for name, para in model.named_parameters():
        if "image_encoder" in name and "attention_prompt" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    # 计时
    timer = utils.Timer()
    # 训练世代循环
    for epoch in range(epoch_start, epoch_max + 1):
        # 在每个epoch之前调用set_epoch（）方法，使得随机打乱顺序在多个epoch能够正常工作
        # 否则每个epoch将采用相同的顺序
        # 在单机单卡下，RandomSampler的seed会在每个epoch随机生成，所以每个epoch采样的数据顺序不同
        # 而在多卡下，为保证每个rank获取数据的indices一致，因此没有改变seed，而通过epoch来手动设置
        train_loader.sampler.set_epoch(epoch)
        # 开始时刻
        t_epoch_start = timer.t()
        # 模型训练
        train_loss_G = train(train_loader, model)
        # 学习率调整
        lr_scheduler.step()

        # 日志信息
        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            # 保存模型
            save(config, model, save_path, 'last')

        # 验证
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                # 保存验证性能最佳的模型
                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    # 模型设置及超参
    parser.add_argument('--config', default="configs/cod-sam-vit-h-attention-prompt.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    # 模型保存路径 名称
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
