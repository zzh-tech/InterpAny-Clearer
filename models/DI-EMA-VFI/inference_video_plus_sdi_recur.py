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
from Trainer_recur import Model
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

    for i, embt in enumerate(embts):
        mid = I0
        pre_mid = I0
        for j in range(args.iters):
            pre_embt = (embt * float(j)) / float(args.iters)
            cur_embt = (embt * (float(j) + 1)) / float(args.iters)
            mid = model.inference(I0, I1, pre_mid, cur_embt, pre_embt)
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
                        default='./experiments/EMA-VFI_m/train_sdi_log/',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='./demo/I0_results/', help='where to save image results')
    parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    parser.add_argument('--iters', type=int, default=2, help='iteration times for inference')
    # downsampling
    parser.add_argument('--skip', type=int, default=1, help='down-sampling factor')
    # half resolution
    parser.add_argument('--half', action='store_true', help='whether to use half resolution')
    parser.add_argument('--no_interp', action='store_true', help='do not interpolate video')
    parser.add_argument('--fps', type=int, default=None, help='fps of the output video')
    args = parser.parse_args()

    model = Model(-1)
    model.load_model(log_path=args.checkpoint)
    model.eval()
    model.device()

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # read video
    if osp.isdir(args.video):
        video_list = [osp.join(args.video, v) for v in os.listdir(args.video) if
                      v.endswith('.mp4') or v.endswith('.mov')]
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
                if args.half:
                    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                gif_imgs.append(frame)
            else:
                break
        cap.release()

        # down-sampling
        gif_imgs = gif_imgs[::args.skip]

        # remove repeated frames
        gif_imgs = [img for i, img in enumerate(gif_imgs) if i == 0 or not np.all(img == gif_imgs[i - 1])]

        if not args.no_interp:
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
        videofourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if args.fps is not None:
            fps = args.fps
        videoWriter = cv2.VideoWriter(save_path, videofourcc, int(fps), (gif_imgs[0].shape[1], gif_imgs[0].shape[0]))
        # count = 0
        # save gif
        for i, img in enumerate(gif_imgs):
            # save image
            # img_path = save_path.replace('.mp4', '_{}.png'.format(i))
            # cv2.imwrite(img_path, img)
            videoWriter.write(img)
            # count += 1
            # print(count)
        videoWriter.release()
