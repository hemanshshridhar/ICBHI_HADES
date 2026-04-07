import math
import torch
import torch.nn.functional as F
import sys
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch import nn
from scipy.ndimage import zoom
import torch.fft
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TSSAEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, dim, dropout=dropout, step_size=ista))
                    ]
                )
            )

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        x1 = (x[:, 0] + x[:, 1]) / 2    
        return x,x1
        
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        nn.init.kaiming_uniform_(self.qkv.weight)
        nn.init.kaiming_uniform_(self.to_out[0].weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):

        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}

# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm, attn_layer=AttentionTSSA,
#                  eta=1e-6):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = attn_layer(
#             dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer, drop=drop)

#         # LayerScale (gamma) params
#         self.gamma1 = nn.Parameter(torch.ones(dim) * eta)
#         self.gamma2 = nn.Parameter(torch.ones(dim) * eta)

#     def forward(self, x, H=None, W=None):
#         x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
#         return x

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dropout = 0.0
        heads = 8
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        nn.init.kaiming_uniform_(self.to_qkv.weight)
        nn.init.kaiming_uniform_(self.to_out[0].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, spatial_size=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_layer=SelfAttention,
                 eta=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim=dim
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # LayerScale (gamma) params
        self.gamma1 = nn.Parameter(torch.ones(dim) * eta)
        self.gamma2 = nn.Parameter(torch.ones(dim) * eta)

    def forward(self, x, H=None, W=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



class TOSTDenoisingHead(nn.Module):
    def __init__(self, feature_dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.attn = AttentionTSSA(
            dim=feature_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=True
        )
        self.residual = True  # Optional: keep residual connection
        self.norm2 = RMSNorm(feature_dim)
    def forward(self, x):
        """
        x: Tensor of shape (B, T, F)
        returns: Filtered Tensor of shape (B, T, F)
        # """
        # x_norm = self.norm2(x)
        x_filtered = self.attn(x)

        if self.residual:
            return x + x_filtered  # Residual denoising
        else:
            return x_filtered


def drop_path_schedule(depth, drop_path_rate):
    # linear schedule 0 -> drop_path_rate
    return [drop_path_rate * i / (depth - 1) for i in range(depth)] if depth > 1 else [drop_path_rate]

# class TSSAEncoder(nn.Module):
#     def __init__(self, dim, depth=6, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.,
#                  drop_path_rate=0.1, eta=1e-6, norm_layer=nn.LayerNorm):
#         super().__init__()
#         dpr = drop_path_schedule(depth, drop_path_rate)
#         self.blocks = nn.ModuleList([
#             Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                   drop=drop, attn_drop=attn_drop, drop_path=dpr[i],
#                   norm_layer=norm_layer, eta=eta)
#             for i in range(depth)
#         ])
#         self.norm = norm_layer(dim)

#     def forward(self, x):
#         # x: (B, T, F=dim)
#         for blk in self.blocks:
#             x = blk(x)
#         x = (x[:, 0] + x[:, 1]) / 2
#         return x

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """
    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class AFF(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(torch.empty(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(torch.zeros(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(torch.empty(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(torch.zeros(2, self.num_blocks, self.block_size))


        nn.init.kaiming_uniform_(self.w1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w2, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        return x + bias



class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """
    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)            # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)




class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    """
    def __init__(self, d_model, num_heads, depth):
        """
        Args:
            d_model (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            depth (float): Initial value for lambda.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        # self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        # self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        # self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        # self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q1 = nn.Parameter(torch.empty(num_heads, self.d_head))
        torch.nn.init.kaiming_uniform_(self.lambda_q1, a=0, mode='fan_in', nonlinearity='relu')

        self.lambda_k1 = nn.Parameter(torch.empty(num_heads, self.d_head))
        torch.nn.init.kaiming_uniform_(self.lambda_k1, a=0, mode='fan_in', nonlinearity='relu')

        self.lambda_q2 = nn.Parameter(torch.empty(num_heads, self.d_head))
        torch.nn.init.kaiming_uniform_(self.lambda_q2, a=0, mode='fan_in', nonlinearity='relu')

        self.lambda_k2 = nn.Parameter(torch.empty(num_heads, self.d_head))
        torch.nn.init.kaiming_uniform_(self.lambda_k2, a=0, mode='fan_in', nonlinearity='relu')
        self.lambda_init = lambda_init_fn(depth)

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.kaiming_uniform_(self.W_q.weight)
        nn.init.kaiming_uniform_(self.W_k.weight)
        nn.init.kaiming_uniform_(self.W_v.weight)
        nn.init.kaiming_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        # print(f"Max value in Q: {Q.max()}")
        # print(f"Min value in Q: {Q.min()}")
        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        # print(f"Max value in V: {V.max()}")
        # print(f"Min value in V: {V.min()}")
        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).clamp_(-30,30).float()  # (num_heads,)
        lambda_q1_dot_k1  = lambda_q1_dot_k1.type_as(Q)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).clamp_(-30, 30).float()  # (num_heads,)
        lambda_q2_dot_k2  = lambda_q2_dot_k2.type_as(Q)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)
        # print(f"Max value in lambda_q1_dot_k1: {lambda_q1_dot_k1.max()}")
        # print(f"Min value in lambda_q1_dot_k1: {lambda_q1_dot_k1.min()}")

        # print(f"Max value in lambda_val: {lambda_val.max()}")
        # print(f"Min value in lambda_val: {lambda_val.min()}")
        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        lambda_val = torch.nan_to_num(lambda_val)
        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A1 = torch.nan_to_num(A1)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.nan_to_num(A2)
        # A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        # A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        # print(f"Max value in A1: {A1.max()}")
        # print(f"Min value in A1: {A1.min()}")
        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)
        # print(f"Max value in attention1: {attention1.max()}")
        # print(f"Min value in attention1: {attention1.min()}")
        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        # print(f"Max value in attention2: {attention2.max()}")
        # print(f"Min value in attention2: {attention2.min()}")
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)

        norm_shape = (N,)
        layer_norm = nn.LayerNorm(norm_shape, elementwise_affine=True).to('cuda:0')


        attention_reshaped = attention.view(-1, *norm_shape)
        attention_normalized = layer_norm(attention_reshaped)
        attention_normalized = attention_normalized.view(attention.shape)




        # print(f"Max value in attention_normalized: {attention_normalized.max()}")
        # print(f"Min value in attention_normalized: {attention_normalized.min()}")
        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)
        # print(f"Max value in V: {V.max()}")
        # print(f"Min value in V: {V.min()}")
        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)
        # Final linear projection

        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out

    def infer_draw(self, X, save_path):
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * num_heads * d_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        # lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        # lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        # lambda_q1_dot_k1 = (self.lambda_q1 * self.lambda_k1).sum(-1).clamp(-CLAMP, CLAMP)
        # lambda_q2_dot_k2 = (self.lambda_q2 * self.lambda_k2).sum(-1).clamp(-CLAMP, CLAMP)
        lambda_q1_dot_k1  = (self.lambda_q1 * self.lambda_k1).sum(-1).clamp(-60, 60).float()
        lambda_q2_dot_k2 = (self.lambda_q2 * self.lambda_k2).sum(-1).clamp(-60, 60).float()
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A1 = torch.nan_to_num(A1)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.nan_to_num(A2)
        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)

        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)
        # Visualize attention heatmap for the first head (index 0)
        self.visualize_attention_heatmaps(attention1, f"{save_path}attn1.png")
        self.visualize_attention_heatmaps(attention2, f"{save_path}attn2.png")
        self.visualize_attention_heatmaps(attention, f"{save_path}attn_all.png")

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)
        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out


    def visualize_attention_heatmaps(self, attention_weights, save_path, draw_beats=False):
        """
        Visualize the attention weights for all heads as heatmaps with consistent color scaling and optional beat overlay.
        Args:
            attention_weights (Tensor): The attention weights of shape (num_heads, sequence_length, sequence_length).
            save_path (str): Path to save the resulting image.
            draw_beats (bool): Whether to draw beat lines on the heatmap (default is True).
        """
        attention_weights = attention_weights[0].detach().cpu().numpy()  # Shape: (num_heads, seq_len, seq_len)
        num_heads = attention_weights.shape[0]

        # Load beat labels
        beat_times = []
        beat_positions = []
        with open("demo/draw_beat/hiphop_00000.beats", 'r') as f:
            for line in f:
                time, position = map(float, line.split())
                if time <= 5:  # Only take beats within sequence duration
                    beat_times.append(time)
                    beat_positions.append(int(position))
                else:
                    break

        # Map beat times to attention positions
        sequence_length = attention_weights.shape[1]
        beat_positions_in_attention = [int(time / 5 * sequence_length) for time in beat_times]

        # Create subplots for each head, arranged in 1 row and 8 columns
        fig, axes = plt.subplots(1, 8, figsize=(90, 10))  # 1 row, 8 columns
        axes = axes.flatten()  # Flatten axes for consistent indexing

        for head in range(num_heads):
            ax = axes[head]
            im = ax.imshow(
                attention_weights[head],
                cmap='viridis',
                aspect='auto',
            )
            # Set title with larger font size
            ax.set_title(f"Head {head + 1}", fontsize=80)

            # Remove x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add a border to each subplot
            for spine in ax.spines.values():
                spine.set_edgecolor('black')  # Set border color
                spine.set_linewidth(2)       # Set border width

            if draw_beats:
            # Overlay beats with different colors for Downbeats and Beats
                for time, pos, beat_pos in zip(beat_times, beat_positions, beat_positions_in_attention):
                    if pos == 1:  # Downbeat
                        ax.axvline(x=beat_pos, color='red', linestyle='--', alpha=0.8)
                    else:  # Regular beat
                        ax.axvline(x=beat_pos, color='blue', linestyle='--', alpha=0.8)

        # Adjust layout
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


# class SelfAttention(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         dropout = 0.0
#         heads = 8
#         dim_head = dim // heads
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x, spatial_size=None):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """
    def __init__(self, d_model, num_heads, depth):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """

        super().__init__()

        self.norm1 = RMSNorm(d_model)

        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, depth)


        self.norm2 = RMSNorm(d_model)

        self.ff = SwiGLU(d_model)

        self.classifier = nn.Linear(256, 4)

        self.fft_filter = AFF(hidden_size=d_model)
        # self.encoder = TSSAEncoder(dim=256, depth=6, num_heads=8, drop=0.0,
        #             attn_drop=0.1, drop_path_rate=0.1, eta=1e-6)

    def forward(self, x):

        x1 = self.fft_filter(x)
        # print(f'output after AFF block {x1.shape}')
        x2 = (x1[:, 0] + x1[:, 1]) / 2    #adding teh real and imaginary part of the
        x2 = self.classifier(x2)
        y = self.attn(self.norm1(x1)) + x1
        # y = self.block(self.norm1(x)) + x

        # Apply SwiGLU Feed-Forward Network with residual connection
        z1 = self.ff(self.norm2(y)) + y
        z = self.fft_filter(z1)
        # x2 = (y[:, 0] + y[:, 1]) / 2    #adding teh real and imaginary part of the
        # x2 = self.classifier(x2)
        # x2 = (z[:, 0] + z[:, 1]) / 2  
        # x2 = self.classifier(x2)
   
        return z, x2


