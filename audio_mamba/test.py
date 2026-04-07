from audio_mamba import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time     

from audio_mamba import AudioMamba


# --- choose device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# Arguments about the data
data_args = Namespace(
    num_mel_bins = 128,
    target_length = 1024,
    mean = -5.0767093,
    std = 4.4533687,
)

# Arguments about the model
model_args = Namespace(
    model_type = 'base',
    n_classes = 4,
    imagenet_pretrain = False,
    imagenet_pretrain_path = None,
    aum_pretrain = True,
    aum_pretrain_path = "/content/drive/MyDrive/MoCo/MVST/16/pretrained_models/base_audioset-vggsound-46.78.pth",
    aum_variant = 'Fo-Bi',
    device = 'cuda',
)


# Embedding dimension
if 'base' in model_args.model_type:
    embed_dim = 768
elif 'small' in model_args.model_type:
    embed_dim = 384
elif 'tiny' in model_args.model_type:
    embed_dim = 192

# AuM block type
bimamba_type = {
    'Fo-Fo': 'none', 
    'Fo-Bi': 'v1', 
    'Bi-Bi': 'v2'
}.get(
    model_args.aum_variant, 
    None
)


# Create the model
AuM = AudioMamba(
    spectrogram_size=(data_args.num_mel_bins, data_args.target_length),
    patch_size=(16, 16),
    strides=(16, 16),
    embed_dim=embed_dim,
    num_classes=model_args.n_classes,
    imagenet_pretrain=model_args.imagenet_pretrain,
    imagenet_pretrain_path=model_args.imagenet_pretrain_path,
    aum_pretrain=model_args.aum_pretrain,
    aum_pretrain_path=model_args.aum_pretrain_path,
    bimamba_type=bimamba_type,
).to(device=device, dtype=dtype)

checkpoint_path = "/content/drive/MyDrive/MoCo/MVST/16/pretrained_models/base_audioset-vggsound-46.78.pth"
ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only =False)

# Some checkpoints are stored as {"model": state_dict, "optimizer": ..., ...}
state_dict = ckpt.get("model", ckpt)

# --- Load weights ---
AuM.load_state_dict(state_dict, strict=False)


print(AuM)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")
    return total, trainable

# Example:
count_parameters(AuM)


x = torch.randn(32,1024,128).to(device=device, dtype=dtype)

y= AuM(x)

print(y.shape)