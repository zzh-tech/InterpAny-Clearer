import os
import cv2
import torch
import random
import numpy as np
import numpy.ma as ma
import os.path as osp
from torch.utils.data import Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_path, sdi_name='dis_index_{}_{}_{}.npy',
                 clip=True, blur=True, use_sdi=False, use_mask=False, triplet=False):
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.data_root = data_path
        self.sdi_name = sdi_name
        self.clip = clip
        self.blur = blur
        self.use_sdi = use_sdi
        self.use_mask = use_mask
        self.image_root = osp.join(self.data_root, 'sequences')
        self.triplet = triplet
        if self.triplet:
            train_fn = osp.join(self.data_root, 'tri_trainlist.txt')
            test_fn = osp.join(self.data_root, 'tri_testlist.txt')
        else:
            train_fn = osp.join(self.data_root, 'sep_trainlist.txt')
            test_fn = osp.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
            self.trainlist = [item for item in self.trainlist if item != '']
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
            self.testlist = [item for item in self.testlist if item != '']
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def crop(self, img0, gt, img1, sdi_map, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x + h, y:y + w, :]
        img1 = img1[x:x + h, y:y + w, :]
        gt = gt[x:x + h, y:y + w, :]
        sdi_map = sdi_map[x:x + h, y:y + w, :]
        return img0, gt, img1, sdi_map

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

    def getimg(self, index):
        imgpath = osp.join(self.image_root, self.meta_data[index])

        if self.triplet:
            imgpaths = [imgpath + '/im{}.png'.format(i) for i in range(1, 4)]
            ind = list(range(3))
        else:
            # RIFEm with Vimeo-Septuplet
            imgpaths = [imgpath + '/im{}.png'.format(i) for i in range(1, 8)]
            ind = list(range(7))
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()

        img0 = cv2.imread(imgpaths[ind[0]])
        gt = cv2.imread(imgpaths[ind[1]])
        img1 = cv2.imread(imgpaths[ind[2]])

        masks_start, masks_end = None, None
        if self.use_mask:
            masks_start_path = osp.join(imgpath, 'im{}_masks.npy'.format(ind[0] + 1))
            masks_start = self.load_mask(masks_start_path)
            masks_end_path = osp.join(imgpath, 'im{}_masks.npy'.format(ind[2] + 1))
            masks_end = self.load_mask(masks_end_path)

        sdi_maps = []
        if self.use_sdi:
            sdi_map_path = osp.join(imgpath, self.sdi_name.format(ind[0], ind[1], ind[2]))
            sdi_maps.append(self.load_sdi_map(sdi_map_path))
            sdi_map_path = osp.join(imgpath, self.sdi_name.format(ind[2], ind[1], ind[0]))
            sdi_maps.append(self.load_sdi_map(sdi_map_path))
        else:
            if self.triplet:
                timestep = 0.5
            else:
                timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)
            sdi_map = np.ones((self.h, self.w, 1), dtype=np.float32) * timestep
            sdi_maps.append(sdi_map)
            sdi_maps.append(1. - sdi_map)

        return img0, gt, img1, sdi_maps, masks_start, masks_end

    def __getitem__(self, index):
        img0, gt, img1, sdi_maps, masks_start, masks_end = self.getimg(index)
        sdi_map = sdi_maps[0]
        # print('sdi_map', sdi_map.shape)
        masks = masks_start
        if self.dataset_name == 'train':
            # temporal augmentation, attention!
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                masks = masks_end
                sdi_map = sdi_maps[-1]
            # mask sdi_map
            if self.use_mask:
                sdi_map = self.mask_sdi_map(masks, sdi_map)
            img0, gt, img1, sdi_map = self.crop(img0, gt, img1, sdi_map, 256, 256)  # RIFEm is 224x224
            # rgb/bgr augmentation
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            # up/down flipping
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                sdi_map = sdi_map[::-1]
            # left/right flipping
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                sdi_map = sdi_map[:, ::-1]
            # random rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                sdi_map = cv2.rotate(sdi_map, cv2.ROTATE_90_CLOCKWISE)[..., np.newaxis]
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
                sdi_map = cv2.rotate(sdi_map, cv2.ROTATE_180)[..., np.newaxis]
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                sdi_map = cv2.rotate(sdi_map, cv2.ROTATE_90_COUNTERCLOCKWISE)[..., np.newaxis]
        else:
            if self.use_mask:
                sdi_map = self.mask_sdi_map(masks, sdi_map)

        if self.clip:
            sdi_map = np.clip(sdi_map, 0, 1)
        if self.blur:
            sdi_map = cv2.GaussianBlur(sdi_map, (9, 9), 0)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        sdi_map = np.ascontiguousarray(sdi_map)
        if len(sdi_map.shape) == 2:
            sdi_map = sdi_map[..., np.newaxis]
        sdi_map = torch.from_numpy(sdi_map).permute(2, 0, 1).float()
        if torch.any(torch.isnan(sdi_map)):
            return self.__getitem__(random.randint(0, len(self) - 1))
        return torch.cat((img0, img1, gt), 0), sdi_map
