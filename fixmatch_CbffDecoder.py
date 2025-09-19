import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import yaml
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from dataset.semicd import SemiCDDataset
from model.semseg.sscdModel_CbffDecoder import SSCDModel


parser = argparse.ArgumentParser(description='Cross-Branch Feature Fusion Decoder for Consistency Regularization-based Semi-Supervised Change Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


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

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)


    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)

    trainset_u = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_pre, previous_best_rec, previous_best_f1, previous_best_iou, previous_best_acc = 0.0, 0.0, 0.0, 0.0, 0.0
    epoch = -1


    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_pre = checkpoint['previous_best_pre']
        previous_best_rec = checkpoint['previous_best_rec']
        previous_best_f1 = checkpoint['previous_best_f1']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']

        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('==> Epoch:{:}, LR:{:.5f}, Previous best Changed Pre:{:.2f}, Rec:{:.2f}, F1:{:.2f}, IoU:{:.2f}, OA:{:.2f}'.format(
                        epoch, optimizer.param_groups[0]['lr'], previous_best_pre, previous_best_rec, previous_best_f1, previous_best_iou, previous_best_acc))


        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)


        for i, ((imgA_x, imgB_x, mask_x),
                (imgA_u_w, imgB_u_w, imgA_u_s, imgB_u_s, ignore_mask, cutmix_box),
                (imgA_u_w_mix, imgB_u_w_mix, imgA_u_s_mix, imgB_u_s_mix, ignore_mask_mix, _)) in enumerate(loader):
            
            imgA_x, imgB_x, mask_x = imgA_x.cuda(), imgB_x.cuda(), mask_x.cuda()
            imgA_u_w, imgB_u_w = imgA_u_w.cuda(), imgB_u_w.cuda()
            imgA_u_s, imgB_u_s = imgA_u_s.cuda(), imgB_u_s.cuda()
            ignore_mask = ignore_mask.cuda()
            cutmix_box = cutmix_box.cuda()
            imgA_u_w_mix, imgB_u_w_mix = imgA_u_w_mix.cuda(), imgB_u_w_mix.cuda()
            imgA_u_s_mix, imgB_u_s_mix = imgA_u_s_mix.cuda(), imgB_u_s_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()


            with torch.no_grad():
                model.eval()

                pred_u_w_mix,_ = model(imgA_u_w_mix, imgB_u_w_mix)
                pred_u_w_mix = pred_u_w_mix.detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            imgA_u_s[cutmix_box.unsqueeze(1).expand(imgA_u_s.shape) == 1] = \
                imgA_u_s_mix[cutmix_box.unsqueeze(1).expand(imgA_u_s.shape) == 1]
            imgB_u_s[cutmix_box.unsqueeze(1).expand(imgB_u_s.shape) == 1] = \
                imgB_u_s_mix[cutmix_box.unsqueeze(1).expand(imgB_u_s.shape) == 1]

            model.train()

            num_lb, num_ulb = imgA_x.shape[0], imgA_u_w.shape[0]
            
            preds_cnn, preds_trans = model(torch.cat((imgA_x, imgA_u_w)),torch.cat((imgB_x, imgB_u_w)))
            pred_x, pred_u_w = preds_cnn.split([num_lb, num_ulb])    # cnn_out
            trans_x, trans_u_w = preds_trans.split([num_lb, num_ulb])   # trans_out
            
            cnn_u_s, trans_u_s = model(imgA_u_s,imgB_u_s)
            
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[cutmix_box == 1]
            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]
            ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask_mix[cutmix_box == 1]

            # sup loss
            loss_x = criterion_l(pred_x, mask_x)*0.5 + criterion_l(trans_x, mask_x)*0.5
            
            # unsup loss
            loss_u_s1 = criterion_u(cnn_u_s, mask_u_w_cutmixed)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed != 255).sum().item()

            loss_u_s2 = criterion_u(trans_u_s, mask_u_w_cutmixed)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed != 255).sum().item()

            # total loss
            loss = (loss_x + loss_u_s1*0.5 + loss_u_s2*0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s1', loss_u_s1.item(), iters)
                writer.add_scalar('train/loss_s2', loss_u_s2.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s1: {:.3f}, Loss s2: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s1.avg,total_loss_s2.avg, total_mask_ratio.avg))
        
        
        pre, rec, f1, iou_class, overall_acc = evaluate(model, valloader, cfg)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Changed Pre: {:.2f}'.format(pre[1]))
            logger.info('***** Evaluation ***** >>>> Changed Rec: {:.2f}'.format(rec[1]))
            logger.info('***** Evaluation ***** >>>> Changed F1: {:.2f}'.format(f1[1]))
            logger.info('***** Evaluation ***** >>>> unChanged IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))
            
            writer.add_scalar('eval/changed_Pre', pre[1], epoch)
            writer.add_scalar('eval/changed_Rec', rec[1], epoch)
            writer.add_scalar('eval/changed_F1', f1[1], epoch)
            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

        is_best = iou_class[1] > previous_best_iou
        if is_best:
            previous_best_pre = pre[1]
            previous_best_rec = rec[1]
            previous_best_f1 = f1[1]
            previous_best_iou = max(iou_class[1], previous_best_iou)
            previous_best_acc = overall_acc


        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_pre': previous_best_pre,
                'previous_best_rec': previous_best_rec,
                'previous_best_f1': previous_best_f1,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
            }


            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


def evaluate(model, loader, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()  # tp+fp+fn
    correct_pixel = AverageMeter() # tp
    total_pixel = AverageMeter()  # tp+fp+fn+tn
    tp_fn_meter = AverageMeter()  # tp+fn
    tp_fp_meter = AverageMeter()  # tp+fp

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            imgA = imgA.cuda()
            imgB = imgB.cuda()

            pred, _ = model(imgA, imgB)
            pred = pred.argmax(dim=1)

            intersection, union, target, tp_fp = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

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




if __name__ == '__main__':
    main()
