import torch
import torch.nn.functional as F
import os.path as osp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model.loss import *
from model.warplayer import warp

from config_recur import *


class Model:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss()
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, log_path='./ckpt', name=None, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }

        if rank <= 0:
            if name is None:
                name = self.name
            self.net.load_state_dict(
                convert(
                    torch.load(osp.join(log_path, '{}.pkl'.format(self.name)))
                )
            )

    def save_model(self, log_path='./ckpt', rank=0):
        if rank == 0:
            torch.save(self.net.state_dict(), osp.join(log_path, '{}.pkl'.format(self.name)))

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''

        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor=1 / down_scale, mode="bilinear", align_corners=False) * (
                    1 / down_scale)
            mask = F.interpolate(mask, scale_factor=1 / down_scale, mode="bilinear", align_corners=False)

            af, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, img_ref, sdi_map=None, ref_sdi_map=None):
        af, mf = self.net.feature_bone(img0, img1, img_ref, ref_sdi_map)
        imgs = torch.cat([img0, img1], dim=1)
        flow, mask = self.net.calculate_flow(imgs, sdi_map, af, mf)
        pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
        return pred

    def update(self, imgs, img_ref, gt, learning_rate=0, training=True, sdi_map=0.5, sdi_map_ref=0.):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            flow, mask, merged, pred = self.net(imgs, img_ref, timestep=sdi_map, timestep_ref=sdi_map_ref)
            loss_l1 = (self.lap(pred, gt)).mean()

            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * 0.5

            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else:
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs, img_ref, timestep=sdi_map, timestep_ref=sdi_map_ref)
                return pred, 0
