import argparse
import os

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.data import test_dataset

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

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
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    with torch.no_grad():
        pbar = tqdm(loader, leave=False, desc='val')

        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = batch['inp']
            gt = batch['gt']
            init_gt = batch['init_gt']

            pred = torch.sigmoid(model.infer(inp))

            resize_pred = F.upsample(pred, size=init_gt.shape[2:], mode='bilinear', align_corners=False)

            result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='save/_cod-sam-vit-h-attention-prompt/config.yaml')
    parser.add_argument('--model', default='./save/_cod-sam-vit-h-attention-prompt/model_epoch_best.pth')
    parser.add_argument('--prompt', default='one')
    parser.add_argument('--test_path', default='../../datasets/test')
    parser.add_argument('--test_save_path', type=str, default='./results')
    parser.add_argument('--test_name', type=str, default='run-1')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    """
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=0)
    """

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    model.eval()

    dataset_path = args.test_path

    with torch.no_grad():
        test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
        for dataset in test_datasets:
            save_path = os.path.join(args.test_save_path, config['model']['name'], dataset)
            #visual_path = os.path.join(args.visual_save_path, dataset)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            #if not os.path.exists(visual_path):
                #os.makedirs(visual_path)
            image_root = os.path.join(dataset_path, dataset, 'image')
            gt_root = os.path.join(dataset_path, dataset, 'mask')
            test_loader = test_dataset(image_root, gt_root, config['test_dataset']['wrapper']['args']['inp_size'])
            for i in range(test_loader.size):
                image, gt, init_gt, name = test_loader.load_data()
                #gt = np.asarray(gt, np.float32)
                #gt /= (gt.max() + 1e-8)
                image = image.cuda()
                gt = gt.unsqueeze(0).cuda()
                init_gt = init_gt.squeeze()

                # test second branch
                #pred = torch.sigmoid(model.infer(image, gt))
                pred = torch.sigmoid(model.infer(image))
                init_pred = pred.data.cpu().numpy().squeeze()

                resize_pred = F.upsample(pred, size=init_gt.shape, mode='bilinear', align_corners=False)
                resize_pred = resize_pred.data.cpu().numpy().squeeze()
                resize_pred = (resize_pred - resize_pred.min()) / (resize_pred.max() - resize_pred.min() + 1e-8)
                #print('save img to: ', os.path.join(save_path, name))
                cv2.imwrite(os.path.join(save_path, name), resize_pred * 255)


    """
    metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    """
