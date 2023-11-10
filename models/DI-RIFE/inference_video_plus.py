import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import os
import cv2
import torch
import os.path as osp
import numpy as np
from model.RIFE_m import Model
from argparse import ArgumentParser
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate(I0, I1, num):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    timesteps = [j / (num + 1 + 1e-6) for j in range(1, num + 1)]

    n, c, h, w = I0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    I0 = F.pad(I0, padding)
    I1 = F.pad(I1, padding)

    for i, timestep in enumerate(timesteps):
        mid = model.inference(I0, I1, timestep=timestep)[0]
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid[:h, :w])
    return imgs


if __name__ == '__main__':
    parser = ArgumentParser()
    # video path list as input
    parser.add_argument('--video', type=str, default='../demo/test_videos/', help='path of video or folder of videos')
    parser.add_argument('--checkpoint', type=str, default='./experiments/rife_m/train_m_log_official',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='../demo/test_videos/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    parser.add_argument('--skip', type=int, default=1, help='down-sampling factor')
    parser.add_argument('--half', action='store_true', help='use half resolution')
    args = parser.parse_args()

    model = Model()
    model.load_model(args.checkpoint)
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

    # print(video_list)

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
                if args.half:
                    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                gif_imgs.append(frame)
            else:
                break
        cap.release()

        # down-sampling
        gif_imgs = gif_imgs[::args.skip]

        print('Before interpolate, the video has {} frames'.format(len(gif_imgs)))
        # interpolate
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
