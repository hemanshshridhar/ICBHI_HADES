import os
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from collections import OrderedDict
import cv2
import PIL
import tqdm
from PIL import Image
import os
import sys
import argparse

from torch import Tensor
from torch.nn.modules import Module
from functools import partial
from typing import Callable, Tuple, Union, Tuple, Union, Any
from collections import defaultdict
from models_DASS.ast_models import DASS



class AttnMamba:
    @staticmethod
    def convert_state_dict_from_mmdet(state_dict):
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.startswith("backbone."):
                new_state_dict[k[len("backbone."):]] = state_dict[k]
        return new_state_dict

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    @torch.no_grad()
    def attnmap_mamba(regs, mode="CB", ret="all", absnorm=0, scale=1, verbose=False, device=None):
        printlog = print if verbose else lambda *args, **kwargs: None
        print(f"attn for mode={mode}, ret={ret}, absnorm={absnorm}, scale={scale}", flush=True)

        _norm = lambda x: x
        if absnorm == 1:
            _norm = lambda x: ((x - x.min()) / (x.max() - x.min()))
        elif absnorm == 2:
            _norm = lambda x: (x.abs() / x.abs().max())

        As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
        print(f'shape of Bs {Bs.shape}')
        us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
        ys, oy = regs["ys"], regs["y"]
        H, W = regs["H"], regs["W"]
        printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
        B, G, N, L = Bs.shape
        GD, N = As.shape
        
        D = GD // G
        # H, W = int(math.sqrt(L)), int(math.sqrt(L))
        if device is not None:
            As, Bs, Cs, Ds, us, dts, delta_bias, ys, oy = As.to(device), Bs.to(device), Cs.to(device), Ds.to(device), us.to(device), dts.to(device), delta_bias.to(device), ys.to(device), oy.to(device)

        mask = torch.tril(dts.new_ones((L, L)))
        dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
        dw_logs = As.view(G, D, N)[None, :, :, :, None] * dts[:,:,:,None,:] # (B, G, D, N, L)
        ws = torch.cumsum(dw_logs, dim=-1).exp()

        if mode == "CB":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:]
        elif mode == "CBdt":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:] * dts.view(B, G, D, 1, L)
        elif mode == "CwBw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:] / ws.clamp(min=1e-20)
        elif mode == "CwBdtw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:]  * dts.view(B, G, D, 1, L) / ws.clamp(min=1e-20)
        elif mode == "ww":
            Qs, Ks = ws, 1 / ws.clamp(min=1e-20)
        else:
            raise NotImplementedError

        printlog(ws.shape, Qs.shape, Ks.shape)
        printlog("Bs", Bs.max(), Bs.min(), Bs.abs().min())
        printlog("Cs", Cs.max(), Cs.min(), Cs.abs().min())
        printlog("ws", ws.max(), ws.min(), ws.abs().min())
        printlog("Qs", Qs.max(), Qs.min(), Qs.abs().min())
        printlog("Ks", Ks.max(), Ks.min(), Ks.abs().min())
        _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L)
        attns = (_Qs.transpose(1, 2) @ _Ks).view(B, G, -1, L, L)
        attns = attns.mean(dim=2) * mask

        attn0 = attns[:, 0, :].view(B, -1, L, L)
        attn1 = attns[:, 1, :].view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)
        attn2 = attns[:, 2, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
        attn3 = attns[:, 3, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
        attn3 = attn3.view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)

        # ao0, ao1, ao2, ao3: attntion in four directions without rearrange
        # a0, a1, a2, a3: attntion in four directions with rearrange
        # a0a2, a1a3, a0a1: combination of "a0, a1, a2, a3"
        # all: combination of all "a0, a1, a2, a3"
        if ret in ["ao0"]:
            attn = _norm(attns[:, 0, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao1"]:
            attn = _norm(attns[:, 1, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao2"]:
            attn = _norm(attns[:, 2, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao3"]:
            attn = _norm(attns[:, 3, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["a0"]:
            attn = _norm(attn0).mean(dim=1)
        elif ret in ["a1"]:
            attn = _norm(attn1).mean(dim=1)
        elif ret in ["a2"]:
            attn = _norm(attn2).mean(dim=1)
        elif ret in ["a3"]:
            attn = _norm(attn3).mean(dim=1)
        elif ret in ["all"]:
            attn = _norm((attn0 + attn1 + attn2 + attn3)).mean(dim=1)
        elif ret in ["nall"]:
            attn = (_norm(attn0) + _norm(attn1) + _norm(attn2) + _norm(attn3)).mean(dim=1) / 4.0
        else:
            raise NotImplementedError(f"{ret} is not allowed")
        attn = (scale * attn).clamp(max=attn.max())
        return attn[0], H, W

    @classmethod
    @torch.no_grad()
    def get_attnmap_mamba(cls, ss2ds, stage=-1, mode="", verbose=False, raw_attn=False, block_id=0, scale=1, device=None):
        mode1 = mode.split("_")[-1]
        mode = mode[:-(len(mode1) + 1)]
        
        absnorm = 0
        tag, mode = cls.checkpostfix("_absnorm", mode)
        absnorm = 2 if tag else absnorm
        tag, mode = cls.checkpostfix("_norm", mode)
        absnorm = 1 if tag else absnorm

        if raw_attn:
            ss2d = ss2ds if not isinstance(ss2ds, list) else ss2ds[stage][block_id]
            regs = getattr(ss2d, "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            return attn

        allrolattn = None
        for k in range(len(ss2ds[stage])):
            regs = getattr(ss2ds[stage][k], "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            L = H * W
            assert attn.shape == (L, L)
            assert attn.max() <= 1
            assert attn.min() >= 0
            rolattn = 0.5 * (attn.cpu() + torch.eye(L))
            rolattn = rolattn / rolattn.sum(-1)
            allrolattn = (rolattn @ allrolattn) if allrolattn is not None else rolattn
        return allrolattn
   
        
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vssm =  DASS(label_dim=4, imagenet_pretrain=False, audioset_pretrain=True, model_size='medium', blur_blocks=set([2,3]), gaussian_blur=False, kernel_size=7, sigma=3, blur_stage =set([2]) ).cuda().eval()
    ckpt_path = "/pretrained_models/DASS/DASS_Bi-PatchMix-CL/bi_patchmix_cl_best_seed5(77.01,47.58,62.29).pth" 
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # get state dict
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
 

    dtype = torch.float16 if device == "cuda" else torch.float32
    missing, unexpected = vssm.load_state_dict(state_dict, strict=True)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    x = np.load("/audio_dataset/spectrogram_data/batch0_img29_label2.npy")
    x = torch.from_numpy(x).to(device=device, dtype=dtype)

    x = x.permute(0,2,1)
    print(x.shape)  
    vssm_path = None
    for name, m in vssm.named_modules():
        if m.__class__.__name__ == "VSSM":
            vssm_path = name
            net = m
            break
    assert vssm_path is not None, "Could not find inner VSSM module inside DASS"
    print("Using VSSM at:", vssm_path)    
    depths = [2,2,20,2]
    for i,block in enumerate(depths):
      for j in range(block):
        stage = i
        block_id = j
        setattr(net.layers[stage].blocks[block_id].op, "__DEBUG__", True)
        ss2d = net.layers[stage].blocks[block_id].op
        _ = vssm(x)
        if stage > 0:
          attn = AttnMamba.get_attnmap_mamba(ss2d, stage=stage, mode="all_norm_CwBw", raw_attn=True, block_id=block_id)
          print("attn:", attn.shape, attn.min().item(), attn.max().item())
          attn_np = attn.cpu().numpy()
          np.save(f"/attention_maps/DASS_BI_PATCHMIX/attention_map_vmamba_{stage}_{block_id}.npy", attn_np)








