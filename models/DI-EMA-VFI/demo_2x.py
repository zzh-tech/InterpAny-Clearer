import os

import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

# I0_path = '../dataset/vimeo_septuplet/sequences/00068/0261/im1.png'
# I2_path = '../dataset/vimeo_septuplet/sequences/00068/0261/im7.png'
# save_dir = './demo/00068/0261_EMA'

I0_path = '../dataset/vimeo_septuplet/sequences/00080/0050/im1.png'
I2_path = '../dataset/vimeo_septuplet/sequences/00080/0050/im7.png'
save_dir = './demo/00080/0050_EMA'

iters = 7

os.makedirs(save_dir, exist_ok=True)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

print(f'=========================Start Generating=========================')

I0 = cv2.imread(I0_path)
I2 = cv2.imread(I2_path)

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

imgs = [I0_, I2_]
while iters != 0:
    imgs_temp = [I0_, ]
    for I_start, I_end in zip(imgs[:-1], imgs[1:]):
        mid = model.inference(I_start, I_end, TTA=TTA, fast_TTA=TTA)
        imgs_temp.append(mid)
        imgs_temp.append(I_end)
    imgs = imgs_temp
    iters -= 1

imgs = [
    (padder.unpad(img)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1]
    for img in imgs
]

mimsave('{}/demo.gif'.format(save_dir), imgs, duration=1000 / 15.)

print(f'=========================Done=========================')
