import sys
import os

import cv2
import torch
import lpips
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf

sys.path.append('.')
sys.path.append('..')

from datetime import datetime
from utils.utils import read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.metrics.niqe import calculate_niqe


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
    CUDA_VISIBLE_DEVICES=4 python vimeo90k.py -c ../cfgs/AMT-S.yaml -p ../pretrained/amt-s.pth -r /mnt/disks/ssd0/dataset/vimeo_triplet/
    """

    parser = argparse.ArgumentParser(
        prog='AMT',
        description='Vimeo90K evaluation',
    )
    parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml')
    parser.add_argument('-p', '--ckpt', default='pretrained/amt-s.pth', )
    parser.add_argument('-r', '--root', default='data/vimeo_triplet', )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_path = args.config
    ckpt_path = args.ckpt
    root = args.root

    network_cfg = OmegaConf.load(cfg_path).network
    network_name = network_cfg.name
    model = build_from_cfg(network_cfg)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    logger = Logger(osp.join(osp.dirname(args.ckpt), 'test_log.txt'))

    with open(osp.join(root, 'tri_testlist.txt'), 'r') as fr:
        file_list = fr.readlines()

    psnr_list = []
    ssim_list = []
    niqe_list = []
    lpips_list = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    for name in file_list:
        name = str(name).strip()
        if (len(name) <= 1):
            continue
        dir_path = osp.join(root, 'sequences', name)
        logger(dir_path + '/im1.png')
        I0 = img2tensor(read(osp.join(dir_path, 'im1.png'))).to(device)
        I1 = cv2.imread(osp.join(dir_path, 'im2.png'))
        I1_tensor = img2tensor(read(osp.join(dir_path, 'im2.png'))).to(device)
        I2 = img2tensor(read(osp.join(dir_path, 'im3.png'))).to(device)
        embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(device)

        mid = model(I0, I2, embt,
                    scale_factor=1.0, eval=True)['imgt_pred']

        psnr = calculate_psnr(mid, I1_tensor)
        ssim = calculate_ssim(mid, I1_tensor)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        mid = mid[0]
        mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.

        # calculate niqe score
        niqe = calculate_niqe(mid[:, :, ::-1] * 255., crop_border=0)
        niqe_list.append(niqe)

        # calculate lpips score
        # mid = mid[:, :, ::-1]  # bgr image
        mid = torch.from_numpy(2 * mid - 1.).permute(2, 0, 1)[None].float().to(
            device
        )  # (1, 3, h, w) value range from [-1, 1]
        I1 = I1 / 255.
        I1 = I1[:, :, ::-1]  # rgb image
        I1 = torch.from_numpy(2 * I1 - 1.).permute(2, 0, 1)[None].float().to(
            device
        )  # (1, 3, h, w) value range from [-1, 1]
        lpips_value = loss_fn_alex.forward(mid, I1).detach().cpu().numpy()
        lpips_list.append(lpips_value)
        logger("Avg PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} NIQE: {:.4f}".format(
            np.mean(psnr), np.mean(ssim), np.mean(lpips_value), np.mean(niqe)
        ))

    logger("Total Avg PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} NIQE: {:.4f}".format(
        np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list), np.mean(niqe_list)
    ))
