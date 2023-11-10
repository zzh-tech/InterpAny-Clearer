import os
import sys

sys.path.append('.')
sys.path.append('..')
import cv2
import torch
import lpips
import math
import argparse
import numpy as np
import os.path as osp
import config as cfg
from datetime import datetime
from torch.nn import functional as F
from basicsr.metrics.niqe import calculate_niqe
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    """
    Logger class to record training log
    """

    def __init__(self, file_path, verbose=True):
        self.verbose = verbose
        self.create_dir(file_path)
        self.logger = open(file_path, 'a+')

    def create_dir(self, file_path):
        dir = osp.dirname(file_path)
        os.makedirs(dir, exist_ok=True)

    def __call__(self, *args, prefix='', timestamp=False):
        if timestamp:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d, %H:%M:%S - ")
        else:
            now = ''
        if prefix == '':
            info = prefix + now
        else:
            info = prefix + ' ' + now
        for msg in args:
            if not isinstance(msg, str):
                msg = str(msg)
            info += msg + '\n'
        self.logger.write(info)
        if self.verbose:
            print(info, end='')
        self.logger.flush()

    def __del__(self):
        self.logger.close()


if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=7 screen python Vimeo90K_sdi_m.py --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/
    CUDA_VISIBLE_DEVICES=2 screen python Vimeo90K_sdi_m.py --testset_path /mnt/disks/ssd0/dataset/vimeo_septuplet/ --unif
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../experiments/EMA-VFI_sdi_m/train_sdi_log/',
                        help='path of checkpoint')
    parser.add_argument('--testset_path', type=str, default='../dataset/vimeo_septuplet/')
    parser.add_argument('--mid_only', action='store_true')
    parser.add_argument('--unif', action='store_true')
    parser.add_argument('--sdi_name', type=str, default='dis_index_{}_{}_{}.npy')
    args = parser.parse_args()

    model = Model(-1)
    model.load_model(log_path=args.checkpoint)
    model.eval()
    model.device()

    if args.unif:
        logger = Logger(osp.join(args.checkpoint, 'unif_test_log.txt'))
    else:
        logger = Logger(osp.join(args.checkpoint, 'test_log.txt'))

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

        gt_indices = [j for j in range(1, 6)]

        if args.unif:
            timesteps = [j / (6 + 1e-6) for j in range(1, 6)]
            timesteps = [torch.tensor(timestep).float().view(1, 1, 1, 1).to(device) for timestep in timesteps]
            embts = timesteps
        else:
            sdi_maps = [np.load(path + 'sequences/' + name + '/{}'.format(
                args.sdi_name.format(0, j, 6)
            )).astype(np.float32) for j in range(1, 6)]
            embts = sdi_maps

        sub_psnr_list = []
        sub_ssim_list = []
        sub_niqe_list = []
        sub_lpips_list = []
        for embt, gt_index in zip(embts, gt_indices):
            if args.mid_only:
                if gt_index != 3:
                    continue

            I1 = Is[gt_index]
            h, w, _ = I1.shape

            if not args.unif:
                embt = np.clip(embt, 0, 1)
                embt = cv2.GaussianBlur(embt, (5, 5), 0)
                embt = cv2.resize(embt, dsize=(w, h), interpolation=cv2.INTER_AREA)[..., np.newaxis]
                embt = np.ascontiguousarray(embt)
                embt = torch.from_numpy(embt).permute(2, 0, 1).to(device)[None]
            else:
                embt = F.interpolate(embt, size=(h, w), mode='bilinear', align_corners=False)

            mid = model.multi_inference(I0, I2, time_list=[embt, ], TTA=False, fast_TTA=False)[0]
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
