import os
import sys

sys.path.append('../../models/DI-RIFE/')
print(os.listdir('../../models/DI-RIFE/'))

import torch
import json
import cv2
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
from model.RIFE_sdi_recur import Model
from scipy.interpolate import interp1d
from torch.nn import functional as F


class Interpolate:
    def __init__(self):
        self.checkpoint = '../../checkpoints/RIFE/DR-RIFE-pro/train_sdi_log'
        self.save_dir = './data/output_recur/'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        self.model = Model()
        self.model.load_model(self.checkpoint)
        self.model.eval()
        self.model.device()

    def interpolate(self, json_dict_path='../webapp/masks_dict.txt', iters=3):
        # remove files under data/output_recur
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for file in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, file))
        # remove files under data/results
        for file in os.listdir('./data/results'):
            os.remove(os.path.join('./data/results', file))
        with open(json_dict_path, 'r') as f:
            mask_dict = json.load(f)
        mask_dict["masks"].reverse()
        masks = np.array(mask_dict["masks"])
        mask_dict["controls"].reverse()
        controls = [np.array(control) for control in mask_dict["controls"]]
        num = int(mask_dict["sampling_points"])

        x = np.linspace(0, 1, num)
        I0_ori = cv2.imread('./data/uploads/' + osp.basename(mask_dict["image1_url"]))
        I1_ori = cv2.imread('./data/uploads/' + osp.basename(mask_dict["image2_url"]))
        I0 = (torch.tensor(I0_ori.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        I1 = (torch.tensor(I1_ori.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        n, c, h, w = I0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)

        a = 0.10
        b = 0.80
        if len(masks) == 1 and np.mean(masks[0]) == 1:
            a = 0.
            b = 1.

        sdi_maps = [torch.zeros_like(I0[:, :1, :, :]) for _ in range(0, num)]
        for i, mask in enumerate(masks):
            control = controls[i]
            ctrl_x = np.linspace(0, 1, len(control))
            ctrl_y = a + b * (np.array(control) / 100.)
            if len(ctrl_x) > 3:
                # when nodes are larger than 3
                interp_func = interp1d(ctrl_x, ctrl_y, kind='cubic')
            elif len(ctrl_x) == 3:
                # when nodes are equal to 3
                interp_func = interp1d(ctrl_x, ctrl_y, kind='quadratic')
            else:
                # when nodes are less than 3
                interp_func = interp1d(ctrl_x, ctrl_y, kind='linear')
            y = interp_func(x)
            for j, sdi_map in enumerate(sdi_maps):
                sdi_maps[j][:, :, mask == 1] = y[j]
        sdi_maps = [transforms.GaussianBlur(kernel_size=9)(sdi_map) for sdi_map in sdi_maps]

        # if the mask is the default one, then return the original start and end image
        if len(masks) == 1 and np.mean(masks[0]) == 1:
            def iter_inference(I0, I1, sdi_maps):
                # iterative inference
                imgs = []
                if len(sdi_maps) == 1:
                    imgs.append(I0)
                elif len(sdi_maps) == 2:
                    imgs.append(I0)
                    imgs.append(I1)
                elif len(sdi_maps) == 3:
                    sdi_map = (sdi_maps[1] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                    imgs.append(I0)
                    imgs.append(self.model.inference(I0, I1, sdi_map=sdi_map, iters=iters))
                    imgs.append(I1)
                elif len(sdi_maps) == 4:
                    imgs.append(I0)
                    sdi_map = (sdi_maps[1] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                    imgs.append(self.model.inference(I0, I1, sdi_map=sdi_map, iters=iters))
                    sdi_map = (sdi_maps[2] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                    imgs.append(self.model.inference(I0, I1, sdi_map=sdi_map, iters=iters))
                    imgs.append(I1)
                else:
                    if len(sdi_maps) % 2 == 1:
                        mid_idx = len(sdi_maps) // 2
                        sid_map = (sdi_maps[mid_idx] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                        I_mid = self.model.inference(I0, I1, sdi_map=sid_map, iters=iters)
                        imgs += iter_inference(I0, I_mid, sdi_maps[:mid_idx + 1])[:-1] + \
                                iter_inference(I_mid, I1, sdi_maps[mid_idx:])
                    else:
                        mid_idx_nxt = len(sdi_maps) // 2
                        mid_idx_pre = mid_idx_nxt - 1
                        sid_map_pre = (sdi_maps[mid_idx_pre] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                        I_mid_pre = self.model.inference(I0, I1, sdi_map=sid_map_pre, iters=iters)
                        sid_map_nxt = (sdi_maps[mid_idx_nxt] - sdi_maps[0]) / (sdi_maps[-1] - sdi_maps[0])
                        I_mid_nxt = self.model.inference(I0, I1, sdi_map=sid_map_nxt, iters=iters)
                        imgs += iter_inference(I0, I_mid_pre, sdi_maps[:mid_idx_pre + 1]) + \
                                iter_inference(I_mid_nxt, I1, sdi_maps[mid_idx_nxt:])
                return imgs

            imgs = iter_inference(I0, I1, sdi_maps)
        else:
            imgs = []
            for sdi_map in sdi_maps:
                imgs.append(self.model.inference(I0, I1, sdi_map=sdi_map, iters=iters))

        imgs = [(img[0].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8) for img in imgs]
        imgs = [img[:h, :w] for img in imgs]

        # create video using cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = osp.join(self.save_dir, 'slomo.mp4')
        # video = cv2.VideoWriter(video_path, fourcc, 15, (w, h))
        os.makedirs(self.save_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            img_file = osp.join(self.save_dir, 'img{}.png'.format(i))
            cv2.imwrite(img_file, img)
            # video.write(img)
        # video.release()
        video_path = video_path.replace('./data', 'http://localhost:5001')
        return video_path


if __name__ == '__main__':
    print("test script")
    interpolator = Interpolate()
    json_dict_path = 'data/mask_dict/masks_dict.txt'
    # json_dict_path = 'data/uploads/55bfaca4-b8bc-4047-9f06-6760659fcde0.txt'
    video_path = interpolator.interpolate(json_dict_path=json_dict_path, num=30, iters=3)
    print(video_path)
    print("end test script")
