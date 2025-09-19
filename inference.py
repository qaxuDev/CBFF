
import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semicd import SemiCDDataset
from model.semseg.sscdModel_CbffDecoder import SSCDModel 

from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from PIL import Image

parser = argparse.ArgumentParser(description='Cross-Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def evaluate(model, loader, result_path, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()  # tp+fp+fn
    correct_pixel = AverageMeter() # tp
    total_pixel = AverageMeter()  # tp+fp+fn+tn
    tp_fn_meter = AverageMeter()  # tp+fn
    tp_fp_meter = AverageMeter()  # tp+fp

    count = 0
    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred, _ = model(imgA, imgB)

            # # 保存预测mask
            out_map = pred[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
            imgname = os.path.join(result_path,id[0])
            Image.fromarray(out_map*255).save(imgname)
            count += 1
            # print('This is the {}th image!'.format(count))
            
            prediction = pred.argmax(dim=1)
            intersection, union, target, tp_fp = intersectionAndUnion(prediction.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            reduced_tp_fp = torch.from_numpy(tp_fp).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)
            dist.all_reduce(reduced_tp_fp)

            intersection_meter.update(reduced_intersection.cpu().numpy())  # tp
            union_meter.update(reduced_union.cpu().numpy())   # tp+fp+fn
            tp_fn_meter.update(reduced_target.cpu().numpy())   # tp+fn
            tp_fp_meter.update(reduced_tp_fp.cpu().numpy())   # tp+fp
            
            correct_pixel.update((pred.cpu() == mask).sum().item()) 
            total_pixel.update(pred.numel())

    pre = intersection_meter.sum / (tp_fp_meter.sum + 1e-10) * 100.0
    rec = intersection_meter.sum / (tp_fn_meter.sum + 1e-10) * 100.0
    f1 = 2 * pre * rec / (pre + rec + 1e-10) 
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    overall_acc = correct_pixel.sum / (total_pixel.sum + 1e-10) * 100.0

    return pre,rec,f1,iou_class, overall_acc


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = SSCDModel(cfg)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)


    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)


    previous_best_pre, previous_best_rec, previous_best_f1, previous_best_iou, previous_best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_pre = checkpoint['previous_best_pre']
        previous_best_rec = checkpoint['previous_best_rec']
        previous_best_f1 = checkpoint['previous_best_f1']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)    
            logger.info('==> Epoch:{:},  best Changed Pre:{:.2f}, Rec:{:.2f}, F1:{:.2f}, IoU:{:.2f}, OA:{:.2f}'.format(
                        epoch, previous_best_pre, previous_best_rec, previous_best_f1, previous_best_iou, previous_best_acc))

    
    result_path = os.path.join(args.save_path,'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    pre, rec, f1, iou_class, overall_acc = evaluate(model, valloader, result_path, cfg)

    if rank == 0:
        logger.info('==> Now best Changed Pre:{:.2f}, Rec:{:.2f}, F1:{:.2f}, IoU:{:.2f}, OA:{:.2f}'.format(
                         pre[1], rec[1], f1[1], iou_class[1], overall_acc))


if __name__ == '__main__':
    main()



