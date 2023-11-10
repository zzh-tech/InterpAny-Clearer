import sys

sys.path.append('..')

import cv2
import math
import torch
import lpips
import argparse
import torchmetrics
import numpy as np
import numpy.ma as ma
import os.path as osp
from model.pytorch_msssim import ssim_matlab
from model.RIFE_sdi_recur import Model
from utils import Logger
from basicsr.metrics.niqe import calculate_niqe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    """
    cmd:
    CUDA_VISIBLE_DEVICES=4 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_sdi_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 1
    CUDA_VISIBLE_DEVICES=5 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_sdi_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 2
    CUDA_VISIBLE_DEVICES=6 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_sdi_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 3
    **************************************************************
    CUDA_VISIBLE_DEVICES=1 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 1
    CUDA_VISIBLE_DEVICES=2 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 2
    CUDA_VISIBLE_DEVICES=3 screen python Vimeo90K_sdi_unif_m_recur.py --model_dir ../experiments/rife_m_recur_noorder_noavg_blur/train_sdi_log --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --sdi_name dis_index_{}_{}_{}.npy --iters 3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='train_sdi_log')
    parser.add_argument('--testset_path', type=str, default='../dataset/vimeo_septuplet/')
    parser.add_argument('--mid_only', action='store_true')
    parser.add_argument('--cont', action='store_true', help='recurrent inference with previous results')
    parser.add_argument('--iters', type=int, default=1, help='iteration times for inference')
    parser.add_argument('--use_flip', action='store_true', help='according sdi value to change the order of input')
    parser.add_argument('--sdi_name', type=str, default='dis_index_{}_{}_{}_avg.npy')
    args = parser.parse_args()

    model = Model(cont=True)
    model.load_model(args.model_dir)
    model.eval()
    model.device()

    if not args.cont:
        if args.use_flip:
            logger = Logger(osp.join(args.model_dir, 'iters{}_flip_unif_test_log.txt'.format(args.iters)))
        else:
            logger = Logger(osp.join(args.model_dir, 'iters{}_test_unif_log.txt'.format(args.iters)))
    else:
        logger = Logger(osp.join(args.model_dir, 'iters{}_cont_test_unif_log.txt'.format(args.iters)))

    path = args.testset_path
    f = open(path + 'sep_testlist.txt', 'r')
    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_list = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    msg = ''
    for i in f:
        name = str(i).strip()
        if (len(name) <= 1):
            continue
        logger(path + 'sequences/' + name + '/im1.png')
        Is = [cv2.imread(path + 'sequences/' + name + '/im{}.png'.format(j)) for j in range(1, 8)]
        h, w, _ = Is[0].shape
        I0 = (torch.tensor(Is[0].transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I2 = (torch.tensor(Is[-1].transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        gt_indices = [j for j in range(1, 6)]
        sdi_maps = [np.load(path + 'sequences/' + name + '/{}'.format(
            args.sdi_name.format(0, j, 6)
        )).astype(np.float32) for j in range(1, 6)]

        sub_psnr_list = []
        sub_ssim_list = []
        sub_lpips_list = []
        sub_niqe_list = []
        last_mid = I0
        last_sdi_map = None
        for sdi_map, gt_index in zip(sdi_maps, gt_indices):
            if args.mid_only:
                if gt_index != 3:
                    continue
            I1 = Is[gt_index]
            h, w, _ = I1.shape
            sdi_map = cv2.resize(sdi_map, dsize=(w, h), interpolation=cv2.INTER_AREA)[..., np.newaxis]
            sdi_map = np.ascontiguousarray(sdi_map)
            sdi_map = torch.from_numpy(sdi_map).permute(2, 0, 1).to(device)[None].float()
            sdi_map = sdi_map * 0. + (gt_index / 6.)

            last_sdi_map = sdi_map
            if (not args.cont):
                if args.use_flip and sdi_map.mean() > 0.5:
                    mid = model.inference(I2, I0, sdi_map=1. - sdi_map, iters=args.iters)
                else:
                    mid = model.inference(I0, I2, sdi_map=sdi_map, iters=args.iters)
            else:
                sdi_map = torch.cat([sdi_map, last_sdi_map, last_mid], dim=1)
                mid = model.inference(I0, I2, sdi_map=sdi_map, iters=0)
            last_mid = mid

            mid = mid[0]
            ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.,
                               torch.round(mid * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
            mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
            I1 = I1 / 255.
            psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
            # print('manual', psnr, ssim)

            sub_psnr_list.append(psnr)
            sub_ssim_list.append(ssim)

            # calculate niqe score
            sub_niqe_list.append(calculate_niqe(mid * 255., crop_border=0))

            # calculate lpips score
            mid = mid[:, :, ::-1]  # rgb image
            mid = torch.from_numpy(2 * mid - 1.).permute(2, 0, 1)[None].float().to(
                device
            )  # (1, 3, h, w) value range from [-1, 1]
            I1 = I1[:, :, ::-1]  # rgb image
            I1 = torch.from_numpy(2 * I1 - 1.).permute(2, 0, 1)[None].float().to(
                device
            )  # (1, 3, h, w) value range from [-1, 1]
            sub_lpips_list.append(loss_fn_alex.forward(mid, I1).detach().cpu().numpy())

        logger("Avg PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} NIQE: {:.4f}".format(
            np.mean(sub_psnr_list), np.mean(sub_ssim_list), np.mean(sub_lpips_list), np.mean(sub_niqe_list)
        ))
        psnr_list += sub_psnr_list
        ssim_list += sub_ssim_list
        lpips_list += sub_lpips_list
        niqe_list += sub_niqe_list

    logger("Total Avg PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} NIQE: {:.4f}".format(
        np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(niqe_list)
    ))
