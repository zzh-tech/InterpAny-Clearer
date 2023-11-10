import sys

sys.path.append('..')

import cv2
import math
import torch
import lpips
import argparse
import numpy as np
import os.path as osp
from model.pytorch_msssim import ssim_matlab
from model.RIFE_m import Model
from utils import Logger
from basicsr.metrics.niqe import calculate_niqe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=0 screen python Vimeo90K_m.py --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='../experiments/rife_m/train_m_log_official/',
                        help='path of checkpoint')
    parser.add_argument('--testset_path', type=str, default='../dataset/vimeo_septuplet/')
    parser.add_argument('--mid_only', action='store_true')
    args = parser.parse_args()

    model = Model()
    model.load_model(args.model_dir)
    model.eval()
    model.device()

    logger = Logger(osp.join(args.model_dir, 'test_log.txt'))

    path = args.testset_path
    f = open(path + 'sep_testlist.txt', 'r')
    psnr_list = []
    ssim_list = []
    niqe_list = []
    lpips_list = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    for i in f:
        name = str(i).strip()
        if (len(name) <= 1):
            continue
        logger(path + 'sequences/' + name + '/im1.png')
        Is = [cv2.imread(path + 'sequences/' + name + '/im{}.png'.format(j)) for j in range(1, 8)]
        I0 = (torch.tensor(Is[0].transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I2 = (torch.tensor(Is[-1].transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        timesteps = [j / (6 + 1e-6) for j in range(1, 6)]
        timesteps = [torch.tensor(timestep).float().view(1, 1, 1, 1).to(device) for timestep in timesteps]
        gt_indices = [j for j in range(1, 6)]
        sub_psnr_list = []
        sub_ssim_list = []
        sub_niqe_list = []
        sub_lpips_list = []
        for timestep, gt_index in zip(timesteps, gt_indices):
            if args.mid_only:
                if gt_index != 3:
                    continue

            mid = model.inference(I0, I2, timestep=timestep)[0]
            I1 = Is[gt_index]
            ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.,
                               mid.unsqueeze(0)).detach().cpu().numpy()
            mid = mid.detach().cpu().numpy().transpose(1, 2, 0)
            # mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
            I1 = I1 / 255.
            psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
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
