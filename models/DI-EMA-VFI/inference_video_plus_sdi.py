import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import os
import cv2
import torch
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
    parser.add_argument('--video', type=str, default='../demo/test_videos/', help='path of video or folder of videos')
    parser.add_argument('--checkpoint', type=str, default='./experiments/EMA-VFI_m/train_sdi_log/',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./demo/I0_results/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[1, 1], help='number of extracted images')
    args = parser.parse_args()

    model = Model(-1)
    model.load_model(log_path=args.checkpoint)
    model.eval()
    model.device()

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # read video
    if osp.isdir(args.video):
        video_list = [osp.join(args.video, v) for v in os.listdir(args.video) if v.endswith('.mp4')]
        video_dir = args.video
    else:
        video_list = [args.video, ]
        video_dir = osp.dirname(args.video)

    for video_path in video_list:
        # cv2 read video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print('fps: {}, h: {}, w: {}'.format(fps, h, w))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = osp.basename(video_path).split('.')[0]
        save_path = video_path.replace(video_dir, args.save_dir)
        # read imgs
        gif_imgs = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                gif_imgs.append(frame)
            else:
                break
        cap.release()

        print('Before interpolate, the video has {} frames'.format(len(gif_imgs)))
        for sub_num in args.num:
            gif_imgs_temp = [gif_imgs[0], ]
            for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
                interp_imgs = interpolate(img_start, img_end, num=sub_num)
                gif_imgs_temp += interp_imgs
                gif_imgs_temp += [img_end, ]
            gif_imgs = gif_imgs_temp
        print('After interpolate, the video has {} frames'.format(len(gif_imgs)))

        # save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(save_path, fourcc, int(fps), (gif_imgs[0].shape[1], gif_imgs[0].shape[0]))
        # count = 0
        for img in gif_imgs:
            videoWriter.write(img)
            # count += 1
            # print(count)
        videoWriter.release()
