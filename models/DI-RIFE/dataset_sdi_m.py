import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32, sdi_name='dis_index_{}_{}_{}.npy', clip=False, blur=False,
                 triplet=True):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.triplet = triplet
        if self.triplet:
            self.data_root = '/mnt/disks/ssd0/dataset/vimeo_triplet'
        else:
            # self.data_root = '/mnt/disks/ssd0/dataset/vimeo_septuplet'
            self.data_root = '../../dataset/vimeo_septuplet'
        self.sdi_name = sdi_name
        self.clip = clip
        self.blur = blur
        self.image_root = os.path.join(self.data_root, 'sequences')
        if self.triplet:
            train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
            test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        else:
            train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
            test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
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
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, img0, gt, img1, sdi_map, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x + h, y:y + w, :]
        img1 = img1[x:x + h, y:y + w, :]
        gt = gt[x:x + h, y:y + w, :]
        sdi_map = sdi_map[x:x + h, y:y + w, :]
        return img0, gt, img1, sdi_map

    def rezie(self, img0, sdi_map):
        h, w, _ = img0.shape
        sdi_map = cv2.resize(sdi_map, dsize=(w, h), interpolation=cv2.INTER_AREA)[..., np.newaxis]
        return sdi_map

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])

        if self.triplet:
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
            ind = [0, 1, 2]
        else:
            # RIFEm with Vimeo-Septuplet
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png',
                        imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
            ind = [0, 1, 2, 3, 4, 5, 6]
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()

        img0 = cv2.imread(imgpaths[ind[0]])
        gt = cv2.imread(imgpaths[ind[1]])
        img1 = cv2.imread(imgpaths[ind[2]])

        # temporal augmentation, attention!
        if self.dataset_name == 'train' and random.uniform(0, 1) < 0.5:
            sdi_map_path = os.path.join(imgpath, self.sdi_name.format(ind[2], ind[1], ind[0]))
            tmp = img1
            img1 = img0
            img0 = tmp
        else:
            sdi_map_path = os.path.join(imgpath, self.sdi_name.format(ind[0], ind[1], ind[2]))
        sdi_map = np.load(sdi_map_path).astype(np.float32)

        if self.clip:
            sdi_map = np.clip(sdi_map, 0, 1)
        if self.blur:
            sdi_map = cv2.GaussianBlur(sdi_map, (5, 5), 0)

        return img0, gt, img1, sdi_map

    def __getitem__(self, index):
        img0, gt, img1, sdi_map = self.getimg(index)
        sdi_map = self.rezie(img0, sdi_map)
        if self.dataset_name == 'train':
            img0, gt, img1, sdi_map = self.crop(img0, gt, img1, sdi_map, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                sdi_map = sdi_map[::-1]
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
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        assert len(sdi_map.shape) == 3
        sdi_map = np.ascontiguousarray(sdi_map)
        sdi_map = torch.from_numpy(sdi_map).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), sdi_map
