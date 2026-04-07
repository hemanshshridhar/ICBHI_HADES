import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial

from models.ast import ASTModel


def _strip_prefix_if_present(state_dict, prefix="module."):
    # Removes "module." if checkpoint was saved from DataParallel/DDP
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _filter_out_heads(state_dict):
    # Keep only backbone weights; drop classification heads if present
    drop_prefixes = (
        "mlp_head.",          # ASTModel's head in many repos
        "classifier.",        # generic
        "head.",              # timm heads
        "fc.",                # common
    )
    return {k: v for k, v in state_dict.items() if not k.startswith(drop_prefixes)}


def load_model_only_weights(model, ckpt_path, device):
    """
    Loads only the model/backbone weights from a checkpoint that may contain
    model/classifier/optimizer/args/etc.
    """
    # If your checkpoint is from YOUR training code and you trust it,
    # simplest is weights_only=False.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Print checkpoint keys (what you wanted earlier)
    if isinstance(ckpt, dict):
        print("Checkpoint top-level keys:", ckpt.keys())
    else:
        print("Checkpoint is not a dict; treating it as a raw state_dict")

    # Extract model state_dict
    if isinstance(ckpt, dict):
        # common patterns:
        #   ckpt["model"] = model.state_dict()
        #   ckpt["state_dict"] = ...
        #   ckpt itself = state_dict
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            # maybe the dict itself is a state_dict
            # (heuristic: tensor-ish values)
            tensorish = any(hasattr(v, "shape") for v in ckpt.values())
            sd = ckpt if tensorish else None
            if sd is None:
                raise ValueError("Could not find model weights in checkpoint dict.")
    else:
        sd = ckpt

    sd = _strip_prefix_if_present(sd, "module.")
    sd = _filter_out_heads(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded model-only weights.")
    print("Missing keys (ok if heads removed):", len(missing))
    print("Unexpected keys:", len(unexpected))
    return model


def main_deit(block_to_extract=8, model_ckpt=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float32
    model = ASTModel(
        label_dim=4,
        fstride=10,           
        tstride=10,           
        input_fdim=128,
        input_tdim=798,       
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
        mix_beta=None
    ).to(device=device, dtype=dtype)

    if model_ckpt is not None:
        model_ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        print(model_ckpt['args'])
        model_ckpt = model_ckpt['model']
        # model.load_state_dict(model_ckpt,strict=False)
        missing, unexpected = model.load_state_dict(model_ckpt, strict=False)
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        # model = load_model_only_weights(model, model_ckpt, device=device)
        print(model)
    model.eval()

    vit = model.v if not hasattr(model, "module") else model.module.v

    attns = {}

    def attn_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)
        setattr(self, "__data__", attn.detach().cpu())
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    for i, blk in enumerate(vit.blocks):
        old_attn = blk.attn
        old_attn._orig_forward = old_attn.forward
        old_attn.forward = partial(attn_forward, old_attn)
        attns[str(i)] = old_attn
    print(attns.keys())

    x = np.load("/audio_dataset/spectrogram_data/batch0_img29_label2.npy")
    print(x.shape)
    x = torch.from_numpy(x).to(device=device, dtype=dtype)
    x = x.unsqueeze(1)  # (B,1,T,F)
    x = F.interpolate(x, size=(798, 128), mode='bilinear', align_corners=False)
    x = x.squeeze(1)    # (B,798,128)

    with torch.no_grad():
        _ = model(x)

    key = str(block_to_extract)
    saved_attn = getattr(attns[key], "__data__", None)
    if saved_attn is None:
        raise RuntimeError(f"No attention saved in block {block_to_extract}.")

    aaa = saved_attn[0]  # (heads, T, T)
    aaa = aaa.mean(dim=0)
    aaa = ((aaa - aaa.min()) / (aaa.max() - aaa.min()))  # (T, T)
    return aaa


def pairwise_cosine_similarity(x):
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    return x @ x.t()


def compute_cosine_similarity(seq_matrix):
    if isinstance(seq_matrix, np.ndarray):
        seq_matrix = torch.tensor(seq_matrix, dtype=torch.float32)
    if seq_matrix.dim() != 2:
        raise ValueError("Input must be 2D.")
    cos_mat = pairwise_cosine_similarity(seq_matrix)
    return cos_mat.mean().item()


if __name__ == "__main__":
    ckpt_path = "/pretrained_models/AST/best.pth"
    layers =11
    for i in range(layers):
      attn_map = main_deit(block_to_extract=i, model_ckpt=ckpt_path)
      print("attn_map shape:", attn_map.shape)
      cos_val = compute_cosine_similarity(attn_map)
      print("Cosine similarity of AST attention:", cos_val)

      np.save(f"/attention_map_ast_{i}.npy",attn_map.numpy())
