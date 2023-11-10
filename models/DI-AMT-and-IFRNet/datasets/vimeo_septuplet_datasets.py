'''
    This code is partially borrowed from IFRNet (https://github.com/ltkong218/IFRNet). 
    In the consideration of the difficulty in flow supervision generation, we abort 
    flow loss in the 8x case.
'''
import os
import cv2
import torch
import random
import numpy as np
import os.path as osp
import numpy.ma as ma
from torch.utils.data import Dataset
from utils.utils import read, img2tensor


def random_resize(img0, imgt, img1, flow=None, sdi_map=None, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        if flow is not None:
            flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
        if sdi_map is not None:
            sdi_map = cv2.resize(sdi_map, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1, flow, sdi_map


def random_crop(img0, imgt, img1, flow=None, sdi_map=None, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x:x + h, y:y + w, :]
    imgt = imgt[x:x + h, y:y + w, :]
    img1 = img1[x:x + h, y:y + w, :]
    if flow is not None:
        flow = flow[x:x + h, y:y + w, :]
    if sdi_map is not None:
        sdi_map = sdi_map[x:x + h, y:y + w]
    return img0, imgt, img1, flow, sdi_map


def random_reverse_channel(img0, imgt, img1, flow=None, sdi_map=None, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1, flow, sdi_map


def random_vertical_flip(img0, imgt, img1, flow=None, sdi_map=None, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        if flow is not None:
            flow = flow[::-1]
            flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
        if sdi_map is not None:
            sdi_map = sdi_map[::-1]
    return img0, imgt, img1, flow, sdi_map


def random_horizontal_flip(img0, imgt, img1, flow=None, sdi_map=None, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        if flow is not None:
            flow = flow[:, ::-1]
            flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
        if sdi_map is not None:
            sdi_map = sdi_map[:, ::-1]
    return img0, imgt, img1, flow, sdi_map


def random_rotate(img0, imgt, img1, flow=None, sdi_map=None, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        if flow is not None:
            flow = flow.transpose((1, 0, 2))
            flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
        if sdi_map is not None:
            sdi_map = sdi_map.transpose((1, 0))
    return img0, imgt, img1, flow, sdi_map


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(self,
                 dataset_dir='/mnt/disks/ssd0/dataset/vimeo_septuplet',
                 augment=True,
                 crop_size=(224, 224),
                 use_flow=False,
                 use_sdi=False,
                 use_mask=False,
                 blur=True,
                 clip=True,
                 sdi_name='dis_index_{}_{}_{}.npy',
                 triplet=False):
        self.data_root = osp.join(dataset_dir, 'sequences')
        self.augment = augment
        self.crop_size = crop_size
        self.use_flow = use_flow
        self.use_sdi = use_sdi
        self.use_mask = use_mask
        self.blur = blur
        self.clip = clip
        self.sdi_name = sdi_name
        self.h = 256
        self.w = 448
        self.triplet = triplet
        if self.triplet:
            train_fn = osp.join(dataset_dir, 'tri_trainlist.txt')
        else:
            train_fn = osp.join(dataset_dir, 'sep_trainlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
            self.trainlist = [item for item in self.trainlist if item != '']
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.triplet:
            self.meta_data = self.trainlist
        else:
            cnt = int(len(self.trainlist) * 0.95)
            self.meta_data = self.trainlist[:cnt]

    def load_sdi_map(self, sdi_map_path):
        sdi_map = np.load(sdi_map_path).astype(np.float32)
        sdi_map = cv2.resize(sdi_map, dsize=(self.w, self.h), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        return sdi_map

    def load_mask(self, masks_path):
        masks_comp = np.load(masks_path)
        masks = np.unpackbits(masks_comp)
        masks = masks.reshape(-1, self.h // 2, self.w // 2).transpose(1, 2, 0)
        masks = cv2.resize(masks.astype(np.uint8), dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        bg_mask = 1 - np.bitwise_or.reduce(masks, axis=-1, keepdims=True)  # add the background mask
        masks = np.concatenate([masks, bg_mask], axis=-1)
        return masks

    def mask_sdi_map(self, masks, sdi_map):
        num_masks = masks.shape[-1]
        masked_sdi_map = ma.array(
            sdi_map.repeat(num_masks, axis=-1),
            mask=1 - masks
        )  # 0 is valid value for numpy.ma, i.e., w/o mask
        mask_avgs = masked_sdi_map.reshape(-1, num_masks).mean(axis=0)
        masked_sdi_map = masks * mask_avgs[np.newaxis, np.newaxis, ...]
        masked_sdi_map = masked_sdi_map.sum(axis=-1, keepdims=True)
        sumed_masks = masks.sum(axis=-1, keepdims=True)
        masked_sdi_map /= sumed_masks
        return masked_sdi_map

    def __getitem__(self, idx):
        img_dir = osp.join(self.data_root, self.meta_data[idx])
        if self.triplet:
            img_paths = [osp.join(img_dir, 'im{}.png'.format(i)) for i in range(1, 4)]
            ind = list(range(3))
        else:
            img_paths = [osp.join(img_dir, 'im{}.png'.format(i)) for i in range(1, 8)]
            ind = list(range(7))
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()

        if random.uniform(0, 1) < 0.5:
            ind.reverse()

        # after temporal augmentation
        if self.triplet:
            embt = 0.5
        else:
            embt = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
        embt = torch.from_numpy(np.array(embt).reshape(1, 1, 1).astype(np.float32))

        img0 = read(img_paths[ind[0]])
        imgt = read(img_paths[ind[1]])
        img1 = read(img_paths[ind[2]])

        if self.use_sdi:
            sdi_map_path = osp.join(img_dir, self.sdi_name.format(ind[0], ind[1], ind[2]))
            sdi_map = self.load_sdi_map(sdi_map_path)
            if self.use_mask:
                masks_path = osp.join(img_dir, 'im{}_masks.npy'.format(ind[0] + 1))
                masks = self.load_mask(masks_path)
                sdi_map = self.mask_sdi_map(masks, sdi_map)
            if self.clip:
                sdi_map = np.clip(sdi_map, 0, 1)
            if self.blur:
                sdi_map = cv2.GaussianBlur(sdi_map, (9, 9), 0)
        else:
            sdi_map = None

        if self.use_flow:
            flow_t0_path = osp.join(img_dir.replace('sequences', 'flow'), 'flow_{}_{}.flo'.format(ind[1], ind[0]))
            flow_t1_path = osp.join(img_dir.replace('sequences', 'flow'), 'flow_{}_{}.flo'.format(ind[1], ind[2]))
            flow_t0 = read(flow_t0_path)
            flow_t1 = read(flow_t1_path)
            flow = np.concatenate((flow_t0, flow_t1), 2).astype(np.float64)
        else:
            flow = None

        if self.augment == True:
            img0, imgt, img1, flow, sdi_map = random_resize(img0, imgt, img1, flow, sdi_map, p=0.1)
            img0, imgt, img1, flow, sdi_map = random_crop(img0, imgt, img1, flow, sdi_map, crop_size=self.crop_size)
            img0, imgt, img1, flow, sdi_map = random_reverse_channel(img0, imgt, img1, flow, sdi_map, p=0.5)
            img0, imgt, img1, flow, sdi_map = random_vertical_flip(img0, imgt, img1, flow, sdi_map, p=0.3)
            img0, imgt, img1, flow, sdi_map = random_horizontal_flip(img0, imgt, img1, flow, sdi_map, p=0.5)
            img0, imgt, img1, flow, sdi_map = random_rotate(img0, imgt, img1, flow, sdi_map, p=0.05)

        img0 = img2tensor(img0.copy()).squeeze(0)
        imgt = img2tensor(imgt.copy()).squeeze(0)
        img1 = img2tensor(img1.copy()).squeeze(0)

        return_dict = {'img0': img0.float(),
                       'imgt': imgt.float(),
                       'img1': img1.float()}
        if self.use_flow:
            flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
            return_dict['flow'] = flow
        if self.use_sdi:
            sdi_map = np.ascontiguousarray(sdi_map)
            if len(sdi_map.shape) == 2:
                sdi_map = sdi_map[..., np.newaxis]
            sdi_map = torch.from_numpy(sdi_map).permute(2, 0, 1).float()
            if torch.any(torch.isnan(sdi_map)):
                return self.__getitem__(random.randint(0, len(self) - 1))
            return_dict['embt'] = sdi_map
        else:
            return_dict['embt'] = embt
        return return_dict


class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self,
                 dataset_dir='/mnt/disks/ssd0/dataset/vimeo_septuplet',
                 crop_size=(224, 224),
                 use_flow=False,
                 use_sdi=False,
                 use_mask=False,
                 blur=True,
                 clip=True,
                 sdi_name='dis_index_{}_{}_{}.npy',
                 triplet=False):
        self.data_root = osp.join(dataset_dir, 'sequences')
        self.crop_size = crop_size
        self.use_flow = use_flow
        self.use_sdi = use_sdi
        self.use_mask = use_mask
        self.blur = blur
        self.clip = clip
        self.sdi_name = sdi_name
        self.h = 256
        self.w = 448
        self.triplet = triplet
        if self.triplet:
            test_fn = osp.join(dataset_dir, 'tri_testlist.txt')
        else:
            test_fn = osp.join(dataset_dir, 'sep_testlist.txt')
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
            self.testlist = [item for item in self.testlist if item != '']
            # print(self.testlist[-3:])
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        self.meta_data = self.testlist

    def load_sdi_map(self, sdi_map_path):
        sdi_map = np.load(sdi_map_path).astype(np.float32)
        sdi_map = cv2.resize(sdi_map, dsize=(self.w, self.h), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        return sdi_map

    def load_mask(self, masks_path):
        masks_comp = np.load(masks_path)
        masks = np.unpackbits(masks_comp)
        masks = masks.reshape(-1, self.h // 2, self.w // 2).transpose(1, 2, 0)
        masks = cv2.resize(masks.astype(np.uint8), dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        bg_mask = 1 - np.bitwise_or.reduce(masks, axis=-1, keepdims=True)  # add the background mask
        masks = np.concatenate([masks, bg_mask], axis=-1)
        return masks

    def mask_sdi_map(self, masks, sdi_map):
        num_masks = masks.shape[-1]
        # print('sdi_map.shape', sdi_map.shape, 'masks.shape', masks.shape)
        masked_sdi_map = ma.array(
            sdi_map.repeat(num_masks, axis=-1),
            mask=1 - masks
        )  # 0 is valid value for numpy.ma, i.e., w/o mask
        mask_avgs = masked_sdi_map.reshape(-1, num_masks).mean(axis=0)
        masked_sdi_map = masks * mask_avgs[np.newaxis, np.newaxis, ...]
        masked_sdi_map = masked_sdi_map.sum(axis=-1, keepdims=True)
        sumed_masks = masks.sum(axis=-1, keepdims=True)
        masked_sdi_map /= sumed_masks
        return masked_sdi_map

    def __getitem__(self, idx):
        img_dir = osp.join(self.data_root, self.meta_data[idx])
        if self.triplet:
            img_paths = [osp.join(img_dir, 'im{}.png'.format(i)) for i in range(1, 4)]
            ind = list(range(3))
        else:
            img_paths = [osp.join(img_dir, 'im{}.png'.format(i)) for i in range(1, 8)]
            ind = list(range(7))
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()

        img0 = read(img_paths[ind[0]])
        imgt = read(img_paths[ind[1]])
        img1 = read(img_paths[ind[2]])

        if self.triplet:
            embt = 0.5
        else:
            embt = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            embt = torch.from_numpy(np.array(embt).reshape(1, 1, 1).astype(np.float32))

        if self.use_sdi:
            sdi_map_path = osp.join(img_dir, self.sdi_name.format(ind[0], ind[1], ind[2]))
            sdi_map = self.load_sdi_map(sdi_map_path)
            if self.use_mask:
                masks_path = osp.join(img_dir, 'im{}_masks.npy'.format(ind[0] + 1))
                masks = self.load_mask(masks_path)
                sdi_map = self.mask_sdi_map(masks, sdi_map)
            if self.clip:
                sdi_map = np.clip(sdi_map, 0, 1)
            if self.blur:
                sdi_map = cv2.GaussianBlur(sdi_map, (9, 9), 0)
        else:
            sdi_map = None

        if self.use_flow:
            flow_t0_path = osp.join(img_dir.replace('sequences', 'flow'), 'flow_{}_{}.flo'.format(ind[1], ind[0]))
            flow_t1_path = osp.join(img_dir.replace('sequences', 'flow'), 'flow_{}_{}.flo'.format(ind[1], ind[2]))
            flow_t0 = read(flow_t0_path)
            flow_t1 = read(flow_t1_path)
            flow = np.concatenate((flow_t0, flow_t1), 2).astype(np.float64)
        else:
            flow = None

        img0 = img2tensor(img0.copy()).squeeze(0)
        imgt = img2tensor(imgt.copy()).squeeze(0)
        img1 = img2tensor(img1.copy()).squeeze(0)

        return_dict = {'img0': img0.float(),
                       'imgt': imgt.float(),
                       'img1': img1.float()}
        if self.use_flow:
            flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
            return_dict['flow'] = flow
        if self.use_sdi:
            sdi_map = np.ascontiguousarray(sdi_map)
            if len(sdi_map.shape) == 2:
                sdi_map = sdi_map[..., np.newaxis]
            sdi_map = torch.from_numpy(sdi_map).permute(2, 0, 1).float()
            if torch.any(torch.isnan(sdi_map)):
                return self.__getitem__(random.randint(0, len(self) - 1))
            return_dict['embt'] = sdi_map
        else:
            return_dict['embt'] = embt
        return return_dict
