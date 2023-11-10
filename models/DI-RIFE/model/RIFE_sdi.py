import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet_sdi import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, local_rank=-1, cont=False, distill=True, flow_smooth=False):
        self.cont = cont
        self.distill = distill
        self.flow_smooth = flow_smooth
        self.flownet = IFNet_sdi(distill=distill)
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6,
                            weight_decay=1e-3)  # use large weight decay may avoid NaN loss
        self.lap = LapLoss()
        if self.flow_smooth:
            self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            model_dict = torch.load('{}/flownet_sdi.pkl'.format(path))
            try:
                self.flownet.load_state_dict(model_dict['model'])
                self.optimG.load_state_dict(model_dict['optimizer'])
            except:
                try:
                    self.flownet.load_state_dict(convert(model_dict['model']))
                    self.optimG.load_state_dict(model_dict['optimizer'])
                except:
                    self.flownet.load_state_dict(convert(model_dict))

    def save_model(self, path, epoch=None, step=None, rank=0):
        if rank == 0:
            model_dict = {
                'model': self.flownet.state_dict(),
                'optimizer': self.optimG.state_dict(),
                'epoch': epoch,
                'step': step
            }
            torch.save(model_dict, '{}/flownet_sdi.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, sdi_map=None):
        if sdi_map is None:
            raise ValueError
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list,
                                                                                      sdi_map=sdi_map)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3),
                                                                                                scale_list,
                                                                                                sdi_map=sdi_map)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, sdi_map, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1),
                                                                                      scale=[4, 2, 1],
                                                                                      sdi_map=sdi_map)
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = loss_l1 * 0.
        if self.distill:
            loss_tea = (self.lap(merged_teacher, gt)).mean()

        loss_smooth = 0.
        if self.flow_smooth:
            loss_smooth = self.sobel(flow[2], flow[2] * 0).mean()

        if training:
            self.optimG.zero_grad()
            if self.distill:
                if self.cont:
                    loss_G = loss_l1 + loss_tea + loss_distill * 0.002  # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
                else:
                    loss_G = loss_l1 + loss_tea + loss_distill * 0.01  # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            else:
                loss_G = loss_l1
            loss_G = loss_G + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
        }
