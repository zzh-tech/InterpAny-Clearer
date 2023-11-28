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
from utils.build_utils import build_from_cfg
from utils.utils import InputPadder
from omegaconf import OmegaConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def interpolate(I0, I1, num):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    padder = InputPadder(I0.shape, 16)
    I0, I1 = padder.pad(I0, I1)

    embts = [torch.zeros_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]

    for i, embt in enumerate(embts):
        mid = I0
        pre_mid = I0
        for j in range(args.iters):
            pre_embt = (embt * float(j)) / float(args.iters)
            cur_embt = (embt * (float(j) + 1)) / float(args.iters)
            mid = model(I0, I1, pre_mid, cur_embt, pre_embt, eval=True)['imgt_pred']
            pre_mid = mid
        mid = padder.unpad(mid)
        mid = mid.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', type=str, default='../demo/test_videos/', help='path of video or folder of videos')
    parser.add_argument('--checkpoint', type=str,
                        default='./experiments/AMT-S_septuplet_wofloloss_400epoch_bs24_lr2e-4/',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./demo/I0_results/', help='where to save image results')
    parser.add_argument('--cfg_name', type=str, default='AMT-S_septuplet_wofloloss.yaml', help='name of config file')
    parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    parser.add_argument('--skip', type=int, default=1, help='down-sampling factor')
    parser.add_argument('--iters', type=int, default=2, help='iteration times for inference')
    parser.add_argument('--fps', type=int, default=None, help='fps of the output video')
    args = parser.parse_args()

    # -----------------------  Load model -----------------------
    cfg_path = osp.join(args.checkpoint, args.cfg_name)
    network_cfg = OmegaConf.load(cfg_path).network
    network_name = network_cfg.name
    ckpt_path = osp.join(args.checkpoint, 'ckpts', 'latest.pth')
    model = build_from_cfg(network_cfg)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    model.eval()

    # -----------------------  Load input frames -----------------------
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
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
        # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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

        # down-sampling
        gif_imgs = gif_imgs[::args.skip]

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
        if args.fps is not None:
            fps = args.fps
        videoWriter = cv2.VideoWriter(save_path, fourcc, int(fps), (gif_imgs[0].shape[1], gif_imgs[0].shape[0]))
        for img in gif_imgs:
            videoWriter.write(img)
        videoWriter.release()
