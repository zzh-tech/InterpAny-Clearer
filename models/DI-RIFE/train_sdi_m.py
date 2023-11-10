import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE_sdi import Model
from dataset_sdi_m import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        # return 2e-4 * mul
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        # return (2e-4 - 2e-6) * mul + 2e-6
        return (3e-4 - 3e-6) * mul + 3e-6


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def train(model, local_rank):
    exp_prefix = './experiments/{}'.format(args.exp_name)
    log_path = '{}/train_sdi_log'.format(exp_prefix)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(exp_prefix, exist_ok=True)
    if local_rank == 0:
        writer = SummaryWriter('{}/train_sdi'.format(exp_prefix))
        writer_val = SummaryWriter('{}/validate_sdi'.format(exp_prefix))
    else:
        writer = None
        writer_val = None
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train', sdi_name=args.sdi_name, clip=args.clip, blur=args.blur, triplet=args.triplet)
    print(
        'load sdi map of training set: {}, clip: {}, blur: {}, triplet: {}'.format(args.sdi_name, args.clip, args.blur,
                                                                                   args.triplet))
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True,
                            sampler=sampler)
    args.step_per_epoch = len(train_data)
    dataset_val = VimeoDataset('validation', sdi_name=args.sdi_name, clip=args.clip, blur=args.blur,
                               triplet=args.triplet)
    print(
        'load sdi map of validation set: {}, clip: {}, blur: {}, triplet: {}'.format(args.sdi_name, args.clip,
                                                                                     args.blur,
                                                                                     args.triplet))
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, sdi_map = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            sdi_map = sdi_map.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * (args.world_size / 4) * (args.batch_size / 16)
            pred, info = model.update(imgs, gt, sdi_map, learning_rate,
                                      training=True)  # pass timestep if you are training RIFEm
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                # writer.add_scalar('loss/tea', info['loss_tea'], step)
                # writer.add_scalar('loss/distill', info['loss_distill'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']),
                                  3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                # merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                # flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for j in range(5):
                    # imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                    imgs = np.concatenate((pred[j], gt[j]), 1)[:, :, ::-1]
                    writer.add_image(str(j) + '/img', imgs, step, dataformats='HWC')
                    # writer.add_image(str(j) + '/flow', np.concatenate((flow2rgb(flow0[j]), flow2rgb(flow1[j])), 1),
                    #                  step, dataformats='HWC')
                    writer.add_image(str(j) + '/flow', flow2rgb(flow0[j]), step, dataformats='HWC')
                    writer.add_image(str(j) + '/mask', mask[j], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                                data_time_interval, train_time_interval,
                                                                                info['loss_l1']))
            step += 1
        nr_eval += 1
        # if nr_eval % 1 == 0:
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank, writer_val)
        model.save_model(log_path, epoch=epoch, step=step, rank=local_rank)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    # loss_distill_list = []
    # loss_tea_list = []
    psnr_list = []
    # psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, sdi_map = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        sdi_map = sdi_map.to(device, non_blocking=True)
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, sdi_map, training=False)
            # merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        # loss_tea_list.append(info['loss_tea'].cpu().numpy())
        # loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            # psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            # psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(10):
                # imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                imgs = np.concatenate((pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    # eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train_sdi_m.py --world_size=1 --batch_size 64 --sdi_name dis_index_{}_{}_{}_avg.npy --exp_name rife_sdi_m --clip --cont
    CUDA_VISIBLE_DEVICES=0,1,2,3 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg --clip --cont
    
    *****************************************************************
    CUDA_VISIBLE_DEVICES=0,1,2,3 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur_wodistill --clip --blur --cont --no_distill
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur_smoothflow --clip --blur --cont --flow_smooth
    CUDA_VISIBLE_DEVICES=0,1,2,3 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29503 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur_wodistill_smoothflow --clip --blur --cont --no_distill --flow_smooth
    
    *****************************************************************
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur_e1200 --clip --blur --cont --epoch 1200
    
    *****************************************************************
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29502 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur_triplet --clip --blur --cont --triplet
    CUDA_VISIBLE_DEVICES=0,1,2,3 screen python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 29504 train_sdi_m.py --world_size=4 --batch_size 16 --sdi_name dis_index_{}_{}_{}.npy --exp_name rife_sdi_m_noavg_blur --clip --blur --cont
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--sdi_name', default='dis_index.npy', type=str, help='file name of sdi map')
    parser.add_argument('--exp_name', default='rife_sdi', type=str, help='file name of sdi map')
    parser.add_argument('--clip', action='store_true', help='whether to clip the value of sdi map')
    parser.add_argument('--blur', action='store_true', help='whether to blur the sdi map')
    parser.add_argument('--cont', action='store_true', help='continuous version')
    parser.add_argument('--triplet', action='store_true', help='whether to use triplet dataset')
    parser.add_argument('--no_distill', action='store_true', help='no optical flow distillation')
    parser.add_argument('--flow_smooth', action='store_true', help='make the flow smooth')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, cont=args.cont, distill=not args.no_distill, flow_smooth=args.flow_smooth)
    train(model, args.local_rank)
