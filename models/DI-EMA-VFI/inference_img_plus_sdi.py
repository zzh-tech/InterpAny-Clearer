import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import os
import cv2
import torch
import imageio as iio
import os.path as osp
import numpy as np
from argparse import ArgumentParser
from Trainer import Model
from benchmark.utils.padder import InputPadder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def interpolate(I0, I1, num):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    padder = InputPadder(I0.shape, 32)
    I0, I1 = padder.pad(I0, I1)

    embts = [torch.zeros_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]

    for i in range(num):
        mid = model.multi_inference(I0, I1, time_list=[embts[i], ], TTA=False, fast_TTA=False)[0]
        mid = padder.unpad(mid)
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img0', type=str, default='./demo/I0_0.png', help='path of start image')
    parser.add_argument('--img1', type=str, default='./demo/I0_1.png', help='path of end image')
    parser.add_argument('--checkpoint', type=str, default='./experiments/EMA-VFI_m/train_sdi_log/',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./demo/I0_results/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    parser.add_argument('--gif', action='store_true', help='whether to generate the corresponding gif')
    args = parser.parse_args()

    extracted_num = 2
    for sub_num in args.num:
        extracted_num += sub_num * (extracted_num - 1)

    # -----------------------  Load model -----------------------
    model = Model(-1)
    model.load_model(log_path=args.checkpoint)
    model.eval()
    model.device()
    # -----------------------  Load input frames -----------------------
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
