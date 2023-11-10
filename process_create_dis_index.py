import os
import sys

sys.path.append('./RAFT/')
sys.path.append('./RAFT/core')
sys.path.append('./data/')

try:
    from data.dis_index import FlowEstimator, cosine_project_ratio
except ImportError:
    from dis_index import FlowEstimator, cosine_project_ratio

import cv2
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
from itertools import combinations

checkpoint = './RAFT/models/raft-things.pth'
sdi_name = 'dis_index'
flow_estimator = FlowEstimator(checkpoint=checkpoint)


def cosine_project_ratio_with_three_imgs(img1_path, img2_path, img3_path, downsample_ratio=2., return_flow=False):
    img1_to_img2, _ = flow_estimator.estimate_flow(img1_path, img2_path)
    img1_to_img3, _ = flow_estimator.estimate_flow(img1_path, img3_path)
    img1_to_img2 = img1_to_img2[0].permute(1, 2, 0).cpu().numpy()
    img1_to_img3 = img1_to_img3[0].permute(1, 2, 0).cpu().numpy()
    img_dis_index = cosine_project_ratio(img1_to_img2, img1_to_img3)
    H, W = img_dis_index.shape
    if downsample_ratio != 1.:
        img_resized_dis_index = cv2.resize(img_dis_index,
                                           dsize=(W // int(downsample_ratio), H // int(downsample_ratio)),
                                           interpolation=cv2.INTER_AREA)
        dis_index = img_resized_dis_index
    else:
        dis_index = img_dis_index
    if return_flow:
        return dis_index, img1_to_img2, img1_to_img3
    else:
        return dis_index


def create_dis_index_for_dataset(sample_paths, avg=False, downsample_ratio=2., sample_length=7):
    for sample_path in tqdm(sample_paths, total=len(sample_paths)):
        sample_path = sample_path.strip()
        img_paths = [osp.join(sample_path, 'im{}.png'.format(i + 1)) for i in range(sample_length)]
        combs = list(combinations(list(range(sample_length)), r=3))

        for comb in combs:
            img1_path = img_paths[comb[0]]
            img2_path = img_paths[comb[1]]
            img3_path = img_paths[comb[2]]
            img_resized_dis_index = cosine_project_ratio_with_three_imgs(img1_path, img2_path, img3_path,
                                                                         downsample_ratio=downsample_ratio)
            img_resized_dis_index_inv = cosine_project_ratio_with_three_imgs(img3_path, img2_path, img1_path,
                                                                             downsample_ratio=downsample_ratio)
            if not avg:
                save_path = osp.join(sample_path, '{}_{}_{}_{}.npy'.format(sdi_name, comb[0], comb[1], comb[2]))
                with open(save_path, 'wb') as f:
                    np.save(f, img_resized_dis_index.astype(np.half))
                save_path = osp.join(sample_path, '{}_{}_{}_{}.npy'.format(sdi_name, comb[2], comb[1], comb[0]))
                with open(save_path, 'wb') as f:
                    np.save(f, img_resized_dis_index_inv.astype(np.half))
            else:
                img_resized_dis_index_avg = (img_resized_dis_index + 1 - img_resized_dis_index_inv) / 2.
                save_path = osp.join(sample_path, '{}_{}_{}_{}_avg.npy'.format(sdi_name, comb[0], comb[1], comb[2]))
                with open(save_path, 'wb') as f:
                    np.save(f, img_resized_dis_index_avg.astype(np.half))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_list_path', type=str)
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--downsample_ratio', type=float, default=2.)
    parser.add_argument('--sample_length', type=int, default=7)
    args = parser.parse_args()
    print('sample_list_path:', args.sample_list_path)
    print('avg:', args.avg)
    with open(args.sample_list_path) as f:
        sample_paths = f.readlines()
    create_dis_index_for_dataset(sample_paths=sample_paths,
                                 avg=args.avg,
                                 downsample_ratio=args.downsample_ratio,
                                 sample_length=args.sample_length)
    # create_dis_index_for_dataset(sample_paths=sample_paths[2::3], avg=args.avg)
