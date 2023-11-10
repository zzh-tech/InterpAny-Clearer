"""
python flow_generation/gen_multi_flow.py -r '../dataset/vimeo_septuplet'
"""
import os
import sys
import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn.functional as F
from itertools import combinations

sys.path.append('.')
from utils.utils import read, write
from flow_generation.liteflownet.run import estimate

parser = argparse.ArgumentParser(
    prog='AMT',
    description='Flow generation',
)
parser.add_argument('-r', '--root', default='/mnt/disks/ssd0/dataset/vimeo_septuplet')
parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()

vimeo90k_dir = args.root
vimeo90k_sequences_dir = osp.join(vimeo90k_dir, 'sequences')
vimeo90k_flow_dir = osp.join(vimeo90k_dir, 'flow')


def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow


print('Built Flow Path')
if not osp.exists(vimeo90k_flow_dir):
    os.makedirs(vimeo90k_flow_dir)

sequences = sorted(os.listdir(vimeo90k_sequences_dir))
for sequences_path in sequences[args.worker_id::args.num_workers]:
    vimeo90k_sequences_path_dir = osp.join(vimeo90k_sequences_dir, sequences_path)
    vimeo90k_flow_path_dir = osp.join(vimeo90k_flow_dir, sequences_path)
    if not osp.exists(vimeo90k_flow_path_dir):
        os.mkdir(vimeo90k_flow_path_dir)

    for sequences_id in sorted(os.listdir(vimeo90k_sequences_path_dir)):
        vimeo90k_flow_id_dir = osp.join(vimeo90k_flow_path_dir, sequences_id)
        if not osp.exists(vimeo90k_flow_id_dir):
            os.mkdir(vimeo90k_flow_id_dir)

for sequences_path in sequences[args.worker_id::args.num_workers]:
    vimeo90k_sequences_path_dir = os.path.join(vimeo90k_sequences_dir, sequences_path)
    vimeo90k_flow_path_dir = os.path.join(vimeo90k_flow_dir, sequences_path)

    for sequences_id in sorted(os.listdir(vimeo90k_sequences_path_dir)):
        vimeo90k_sequences_id_dir = os.path.join(vimeo90k_sequences_path_dir, sequences_id)
        vimeo90k_flow_id_dir = os.path.join(vimeo90k_flow_path_dir, sequences_id)

        img_paths = [osp.join(vimeo90k_sequences_id_dir, 'im{}.png'.format(i + 1)) for i in range(7)]
        combs = list(combinations(list(range(7)), r=3))

        for comb in combs:
            img0_path = img_paths[comb[0]]
            imgt_path = img_paths[comb[1]]
            img1_path = img_paths[comb[2]]

            flow_t0_path = osp.join(vimeo90k_flow_id_dir + '/flow_{}_{}.flo'.format(comb[1], comb[0]))
            flow_t1_path = osp.join(vimeo90k_flow_id_dir + '/flow_{}_{}.flo'.format(comb[1], comb[2]))

            img0 = read(img0_path)
            imgt = read(imgt_path)
            img1 = read(img1_path)

            flow_t0 = pred_flow(imgt, img0)
            flow_t1 = pred_flow(imgt, img1)

            write(flow_t0_path, flow_t0)
            write(flow_t1_path, flow_t1)

        print('{} finished'.format(vimeo90k_sequences_id_dir))

    print('Written Sequences {}'.format(sequences_path))
