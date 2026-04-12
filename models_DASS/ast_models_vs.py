import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '/pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_
import numpy as np
import math


def get_sin_pos(n_position, d_hid, position_rate=1e4):
    pos = torch.arange(0, n_position).unsqueeze(1)
    dim = torch.arange(0, d_hid, 2)
    div_term = torch.exp(dim * -(math.log(position_rate) / d_hid))
    pos = pos.float()
    div_term = div_term.float()
    pos_emb = torch.zeros(n_position, d_hid)
    pos_emb[:, 0::2] = torch.sin(pos * div_term)
    pos_emb[:, 1::2] = torch.cos(pos * div_term)
    return pos_emb


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes
    :param fstride: the stride of patch spliting on the frequency dimension
    :param tstride: the stride of patch spliting on the time dimension
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128,
                 input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False,
                 model_size='base384', verbose=True, pos_emb_type='learned'):

        super(ASTModel, self).__init__()

        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(
                str(imagenet_pretrain), str(audioset_pretrain)))

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224',
                                           pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224',
                                           pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224',
                                           pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384',
                                           pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim,
                                       kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                    1, self.original_num_patches, self.original_embedding_dim
                ).transpose(1, 2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :, :, :,
                        int(self.oringal_hw / 2) - int(t_dim / 2):
                        int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :, :,
                        int(self.oringal_hw / 2) - int(f_dim / 2):
                        int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim,
                        :
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(
                    1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(
                    torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2,
                                self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

            if pos_emb_type == 'sine':
                print("Using sinusoidal positional embedding")
                new_pos_embed = get_sin_pos(num_patches + 2, self.original_embedding_dim)
                new_pos_embed = nn.Parameter(new_pos_embed, requires_grad=False)
                self.v.pos_embed = new_pos_embed

        elif audioset_pretrain:
            if not imagenet_pretrain:
                raise ValueError(
                    'Currently model pretrained on only audioset is not supported, '
                    'please set imagenet_pretrain=True.')
            if model_size != 'base384':
                raise ValueError('Currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url,
                              out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth',
                            map_location=device)
            audio_model = ASTModel(
                label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                imagenet_pretrain=False, audioset_pretrain=False,
                model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[
                    :, :, :, 50 - int(t_dim / 2): 50 - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[
                    :, :, 6 - int(f_dim / 2): 6 - int(f_dim / 2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim,
                              kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1
        batch_size, num_patch, dim = image.size()
        device = image.device
        index = torch.randperm(batch_size).to(device)
        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)
            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)
            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)
            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)
        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    @autocast()
    def forward(self, x, y=None, patch_mix=False):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if patch_mix:
            h_patch = int((x.size()[2] - 16) / 10) + 1
            w_patch = int((x.size()[3] - 16) / 10) + 1
            x, y_a, y_b, lam, index = self.patch_mix(
                x, y, time_domain=False, hw_num_patch=[h_patch, w_patch])
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index


class DASS(nn.Module):
    """
    The DASS model.
    :param label_dim: number of total classes
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use AudioSet pretrained model
    :param enable_patch_mix: enable patch mix augmentation
    :param mix_beta: beta parameter for patch mix
    :param model_size: model size, one of ['small', 'medium']
    :param hades_blocks: set of block indices within hades_stage to apply HADES
    :param hades_stage: set of stage indices to apply HADES
    :param verbose: print model summary
    """
    def __init__(
        self,
        label_dim=527,
        imagenet_pretrain=True,
        audioset_pretrain=False,
        enable_patch_mix=False,
        mix_beta=1.0,
        model_size='medium',
        # hades params (replaces old blur params)
        hades_blocks=None,
        hades_stage=None,
        hades_on=True,
        verbose=True,
    ):
        super(DASS, self).__init__()

        # import the correct vmamba
        if hades_on:
            from .vmamba import VSSM
            print('HADES DASS will start')
        else:
            from .vmamba import VSSM
            print('Normal DASS will start')

        # safe defaults for mutable args
        hades_blocks = set([2, 3, 12]) if hades_blocks is None else set(hades_blocks)
        hades_stage  = set([2])        if hades_stage  is None else set(hades_stage)

        if verbose:
            print('---------------DASS Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(
                str(imagenet_pretrain), str(audioset_pretrain)))

        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.mix_beta = mix_beta
        self.hades_on = hades_on

        if model_size == 'small':
            self.v = VSSM(
                drop_path_rate=0.2, dims=96, depths=[2, 2, 8, 2],
                ssm_d_state=1, ssm_ratio=1.0, ssm_conv_bias=False,
                forward_type='v05_noz', downsample_version="v3",
                patchembed_version="v2", norm_layer="LN2D", num_classes=1000,
                hades_blocks=hades_blocks if hades_on else set(),
                hades_stage=hades_stage  if hades_on else set(),
            )
            if imagenet_pretrain:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://github.com/MzeroMiko/VMamba/releases/download/'
                        '%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth',
                    map_location="cpu", check_hash=True
                )
                self.v.load_state_dict(checkpoint["model"], strict=False)

        elif model_size == 'medium':
            self.v = VSSM(
                drop_path_rate=0.3, dims=96, depths=[2, 2, 20, 2],
                ssm_d_state=1, ssm_ratio=1.0, ssm_conv_bias=False,
                forward_type='v05_noz', downsample_version="v3",
                patchembed_version="v2", norm_layer="LN2D", num_classes=1000,
                hades_blocks=hades_blocks if hades_on else set(),
                hades_stage=hades_stage  if hades_on else set(),
            )
            if imagenet_pretrain:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url='https://github.com/MzeroMiko/VMamba/releases/download/'
                        '%23v2cls/vssm1_small_0229s_ckpt_epoch_240.pth',
                    map_location="cpu", check_hash=True
                )
                self.v.load_state_dict(checkpoint["model"], strict=False)
        else:
            raise Exception('Model size must be one of small, medium.')

        # replace classifier head
        self.final_feat_dim = self.v.classifier.head.in_features
        self.v.classifier.head = nn.Linear(self.v.classifier.head.in_features, label_dim)

        # replace patch embed first conv for single-channel audio input
        new_proj = torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        if imagenet_pretrain:
            new_proj.weight = torch.nn.Parameter(
                torch.sum(self.v.patch_embed[0].weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed[0].bias
        self.v.patch_embed[0] = new_proj
        del new_proj

        # load audioset pretrained weights if requested
        if audioset_pretrain:
            if model_size == 'small':
                checkpoint_path = '/pretrained_models/DASS_small.pth'
            elif model_size == 'medium':
                checkpoint_path = 'pretrained_models/DASS_medium_v2.pth'
                print("=========loaded========")

            checkpoint = torch.load(checkpoint_path)
            mod_checkpoint = {}
            for k, v in checkpoint.items():
                if 'classifier.head' in k:
                    continue
                mod_checkpoint[k.replace('module.v.', '')] = v
            self.v.load_state_dict(mod_checkpoint, strict=False)

    @autocast()
    def forward(self, x):
        """
        :param x: input spectrogram (batch_size, time_frame_num, frequency_bins)
        :param y: labels (needed for patch_mix)
        :param patch_mix: whether to apply patch mix augmentation
        :param mix_type: type of patch mix ('2d', 'time', 'freq')
        :return:
            - no patch_mix + hades_on:  (logits, lb_loss, div_loss)
            - no patch_mix + no hades:  (logits, 0.0, 0.0)
            - patch_mix:                (features, y_a, y_b, lam, index)
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        result = self.v(x)

        logits, lb_loss, div_loss = result
        return logits, lb_loss, div_loss


if __name__ == '__main__':
    # test normal forward
    model = DASS(label_dim=4, imagenet_pretrain=False, audioset_pretrain=False,
                 model_size='medium', hades_on=True)
    print(model)
    print("final_feat_dim:", model.final_feat_dim)

    # test forward pass
    dummy_input = torch.randn(2, 1024, 128)
    dummy_labels = torch.zeros(2, 4)

