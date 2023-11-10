import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from Trainer_recur import Model
from dataset_sdi_m_mask_recur import VimeoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config_recur import *

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5


def train(model, local_rank, batch_size, data_path):
    exp_prefix = './experiments/{}'.format(args.exp_name)
    log_path = '{}/train_sdi_log'.format(exp_prefix)
    os.makedirs(exp_prefix, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    if local_rank == 0:
        writer = SummaryWriter('{}/train_sdi'.format(log_path))
        writer_val = SummaryWriter('{}/validate_sdi'.format(log_path))
    else:
        writer = None
        writer_val = None
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train', data_path, use_sdi=args.use_sdi, use_mask=args.use_mask)
    print('use_sdi: {}, use_mask: {}'.format(args.use_sdi, args.use_mask))
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True,
                            sampler=sampler)
    args.step_per_epoch = len(train_data)
    dataset_val = VimeoDataset('test', data_path, use_sdi=args.use_sdi, use_mask=args.use_mask)
    print('use_sdi: {}, use_mask: {}'.format(args.use_sdi, args.use_mask))
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, data_ref_gpu, sdi_map, sdi_map_ref = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            data_ref_gpu = data_ref_gpu.to(device, non_blocking=True) / 255.
            sdi_map = sdi_map.to(device, non_blocking=True)
            sdi_map_ref = sdi_map_ref.to(device, non_blocking=True)
            imgs, gt = data_gpu[:, :6], data_gpu[:, 6:]
            img_ref = data_ref_gpu
            learning_rate = get_learning_rate(step) * (args.world_size / 4.) * (args.batch_size / 16.)
            _, loss = model.update(imgs, img_ref, gt, learning_rate, sdi_map=sdi_map, sdi_map_ref=sdi_map_ref,
                                   training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, i, args.step_per_epoch,
                                                                             data_time_interval, train_time_interval,
                                                                             loss))
            step += 1
            # if step == 100:
            #     break
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, nr_eval, local_rank, writer_val)
        model.save_model(log_path=log_path, rank=local_rank)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    psnr = []
    for _, data in enumerate(val_data):
        data_gpu, data_ref_gpu, sdi_map, sdi_map_ref = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        data_ref_gpu = data_ref_gpu.to(device, non_blocking=True) / 255.
        sdi_map = sdi_map.to(device, non_blocking=True)
        sdi_map_ref = sdi_map_ref.to(device, non_blocking=True)
        imgs, gt = data_gpu[:, 0:6], data_gpu[:, 6:]
        img_ref = data_ref_gpu
        with torch.no_grad():
            pred, _ = model.update(imgs, img_ref, gt, sdi_map=sdi_map, sdi_map_ref=sdi_map_ref, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))

    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print('epoch:{}, psnr: {:.4f}'.format(nr_eval, psnr))
        writer_val.add_scalar('psnr', psnr, nr_eval)


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_sdi_m_mask_recur.py --world_size 4 --batch_size 8 --exp_name EMA-VFI_sdi_m_recur --use_sdi
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_sdi_m_mask_recur.py --world_size 4 --batch_size 8 --exp_name EMA-VFI_sdi_m_mask_recur --use_sdi --use_mask
    ***********************************************:
    CUDA_VISIBLE_DEVICES=4,5,6,7 screen python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 train_sdi_m_mask_recur.py --world_size 4 --batch_size 8 --exp_name EMA-VFI_m_recur
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--data_path', type=str, default='../../dataset/vimeo_septuplet',
                        help='data path of vimeo90k')
    parser.add_argument('--use_sdi', action='store_true', help='whether to use sdi')
    parser.add_argument('--use_mask', action='store_true', help='whether to use mask')
    parser.add_argument('--exp_name', default='ema-vfi', type=str, help='file name of sdi map')
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    train(model, args.local_rank, args.batch_size, args.data_path)
