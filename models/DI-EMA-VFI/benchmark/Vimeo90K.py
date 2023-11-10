import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import lpips
import warnings
from datetime import datetime
from tqdm import tqdm
import os.path as osp
from basicsr.metrics.niqe import calculate_niqe

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
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


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, default='/mnt/disks/ssd0/dataset/vimeo_triplet')
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'

'''==========Model setting=========='''
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 2, 2]
    )
else:
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

logger = Logger(osp.join('ckpt', 'test_log.txt'))

print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K   Model: {model.name}   TTA: {TTA}')
path = args.path
f = open(path + '/tri_testlist.txt', 'r')
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
    I0 = cv2.imread(path + '/sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + '/sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + '/sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    mid = model.inference(I0, I2, TTA=TTA, fast_TTA=TTA)[0]
    ssim = ssim_matlab(torch.tensor(I1.transpose(2, 0, 1)).cuda().unsqueeze(0) / 255.,
                       mid.unsqueeze(0)).detach().cpu().numpy()
    mid = mid.detach().cpu().numpy().transpose(1, 2, 0)
    I1 = I1 / 255.
    psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    # calculate niqe score
    niqe = calculate_niqe(mid * 255., crop_border=0)
    niqe_list.append(niqe)

    # calculate lpips score
    mid = mid[:, :, ::-1]  # rgb image
    mid = torch.from_numpy(2 * mid - 1.).permute(2, 0, 1)[None].float().to(
        device
    )  # (1, 3, h, w) value range from [-1, 1]
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
