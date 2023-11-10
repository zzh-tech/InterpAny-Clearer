import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet_sdi_recur import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from torch.nn.utils import clip_grad_norm_

# torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, local_rank=-1, cont=False, distill=True, extra=False):
        self.cont = cont
        self.distill = distill
        self.extra = extra
        # if local_rank <= 0:
        #     print('cont: {}, distill: {}, extra: {}'.format(self.cont, self.distill, self.extra))
        self.flownet = IFNet_sdi(distill=self.distill)
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6,
                            weight_decay=1e-3)  # use large weight decay may avoid NaN loss
        # self.optimG = AdamW(self.flownet.parameters(), lr=1e-6,
        #                     weight_decay=1e-2)  # use large weight decay may avoid NaN loss
        self.lap = LapLoss()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
            # self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank,
            #                    find_unused_parameters=True)

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

    @torch.no_grad()
    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], iters=0, sdi_map=None,
                  ref_img=None, ref_sdi_map=None):
        def _inference(img0, img1, scale=1, scale_list=[4, 2, 1], sdi_map=None):
            for i in range(3):
                scale_list[i] = scale_list[i] * 1.0 / scale
            imgs = torch.cat((img0, img1), 1)

            if self.distill:
                flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list,
                                                                                              sdi_map=sdi_map)
            else:
                flow, mask, merged = self.flownet(imgs, scale_list, sdi_map=sdi_map)
            return merged[2]

        if sdi_map is None:
            raise ValueError

        if iters == 0:
            return _inference(img0, img1, scale=scale, scale_list=scale_list, sdi_map=sdi_map)
        else:
            if (ref_sdi_map is not None) and (ref_img is not None):
                sdi_maps = [sdi_map * i / iters + ref_sdi_map * (iters - i) / iters for i in range(iters + 1)]
                img_cur = ref_img
            else:
                sdi_maps = [sdi_map * i / (iters) for i in range(iters + 1)]
                img_cur = img0
            for sdi_map_tgt, sdi_map_pre in zip(sdi_maps[1:], sdi_maps[:-1]):
                sdi_map = torch.cat([sdi_map_tgt, sdi_map_pre, img_cur], dim=1)
                img_pred = _inference(img0, img1, scale=scale, scale_list=scale_list, sdi_map=sdi_map)
                img_cur = img_pred
            return img_pred

    def update(self, imgs, gt, sdi_map, learning_rate=0, mul=1, training=True, flow_gt=None, paths=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        loss_tea = torch.tensor(0.).to(device)
        loss_distill = torch.tensor(0.).to(device)
        if self.distill:
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1),
                                                                                          scale=[4, 2, 1],
                                                                                          sdi_map=sdi_map)
            loss_tea = (self.lap(merged_teacher, gt)).mean()
        else:
            flow, mask, merged = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1], sdi_map=sdi_map)
            merged_teacher = merged[2]
            flow_teacher = flow[2][:, :2]
        loss_l1 = (self.lap(merged[2], gt)).mean()

        loss_G = loss_l1
        if training:
            if self.distill:
                loss_G = loss_G + loss_tea
                if self.cont:
                    loss_G = loss_G + loss_distill * 0.002  # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
                else:
                    loss_G = loss_G + loss_distill * 0.01  # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            self.optimG.zero_grad()
            loss_G.backward()
            # with torch.autograd.detect_anomaly():
            #     loss_G.backward()
            if self.extra:
                clip_grad_norm_(self.flownet.parameters(), max_norm=2.0, norm_type=2)  # enable it for extrapolation
            self.optimG.step()

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
