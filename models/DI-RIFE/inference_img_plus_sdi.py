import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import shutil
import os
import math
import cv2
import torch
import lpips
import imageio as iio
import os.path as osp
import numpy as np
import numpy.ma as ma
from model.RIFE_sdi import Model
from argparse import ArgumentParser
from tqdm import tqdm
from model.pytorch_msssim import ssim_matlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate(I0, I1, num):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    sdi_maps = [torch.zeros_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]

    for i, sdi_map in enumerate(sdi_maps):
        mid = model.inference(I0, I1, sdi_map=sdi_map)[0]
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img0', type=str, default='./demo/I0.png', help='path of start image')
    parser.add_argument('--img1', type=str, default='./demo/I0_1.png', help='path of end image')
    parser.add_argument('--checkpoint', type=str, default='./experiments/rife_sdi_m_mask_noavg_blur/train_sdi_log',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./demo/I0_results/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    parser.add_argument('--gif', action='store_true', help='whether to generate the corresponding gif')
    parser.add_argument('--no_distill', action='store_true', help='no optical flow distillation')
    args = parser.parse_args()

    extracted_num = 2
    for sub_num in args.num:
        extracted_num += sub_num * (extracted_num - 1)

    model = Model(distill=not args.no_distill)
    model.load_model(args.checkpoint)
    model.eval()
    model.device()

    os.makedirs(args.save_dir, exist_ok=True)
    I0 = cv2.imread(args.img0)
    I1 = cv2.imread(args.img1)
    gif_imgs = [I0, I1]

    for sub_num in args.num:
        gif_imgs_temp = [gif_imgs[0], ]
        for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
            interp_imgs = interpolate(img_start, img_end, num=sub_num)
            gif_imgs_temp += interp_imgs
            gif_imgs_temp += [img_end, ]
        gif_imgs = gif_imgs_temp

    print('Interpolate 2 images to {} images'.format(extracted_num))

    for i, img in enumerate(gif_imgs):
        save_path = osp.join(args.save_dir, '{:03d}.png'.format(i))
        cv2.imwrite(save_path, img)

    if args.gif:
        gif_path = osp.join(args.save_dir, 'demo.gif')
        with iio.get_writer(gif_path, mode='I') as writer:
            for img in gif_imgs:
                writer.append_data(img[:, :, ::-1])
