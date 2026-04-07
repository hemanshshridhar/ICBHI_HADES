import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


""" train util """
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def adjust_lr_wd(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    wd = args.weight_decay
    if args.wd_scheduler:
        wd_min = args.weight_decay_end
        wd = wd_min + (wd - wd_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if i == 0: # in case of DINO and ViT, only wd for regularized params
            param_group['weight_decay'] = wd


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(args, optim_params=None, model=None):
    if model is not None:
        optim_params = model.parameters()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(optim_params,
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(optim_params,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise NotImplemented

    return optimizer


class MA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ma_beta, current_model, ma_ckpt):
    ma_updater = MA(ma_beta)
    new_state_dict = {}
    for (k1, current_params), (k2, ma_params) in zip(current_model.state_dict().items(), ma_ckpt.items()):
        assert k1 == k2
        old_weight, up_weight = ma_params.data, current_params.data
        new_state_dict[k1] = ma_updater.update_average(old_weight, up_weight)

    current_model.load_state_dict(new_state_dict)
    return current_model


""" eval util """
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        n_cls = output.shape[1]
        valid_topk = [k for k in topk if k <= n_cls]
        
        maxk = max(valid_topk)
        bsz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k in valid_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / bsz))
            else: res.append(torch.tensor([0.]))

        return res, bsz


def save_model(model,optimizer, args, epoch, save_file):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }

    torch.save(state, save_file)
    del state
    
    
def update_json(exp_name, acc, path='./save/results.json'):
    acc = [round(a, 2) for a in acc]
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)

    with open(path, 'r', encoding="UTF-8") as f:
        result_dict = json.load(f)
        result_dict[exp_name] = acc
    
    with open(path, 'w') as f:
        json.dump(result_dict, f)
        
    print('best Score: {} (sp, se, sc)'.format(acc))        
    print('results updated to %s' % path)
import torch
import numpy as np


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def generate_raw_attn(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    all_layer_attentions = []
    cls_pos = 1024//2
    for layeridx in range(len(model.layers)):
        attn_heads = model.layers[layeridx].mixer.xai_b
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    p = torch.cat(all_layer_attentions[start_layer:], dim=0).mean(dim=0).unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits


def generate_mamba_attr(model, image, start_layer=15):
    image.requires_grad_()
    logits = model(image)
    index = np.argmax(logits.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    all_layer_attentions = []
    cls_pos = 1024//2
    for layeridx in range(len(model.layers)):
        attn_heads = model.layers[layeridx].mixer.xai_b.clamp(min=0)
        s = model.layers[layeridx].get_gradients().squeeze().detach() #[1:, :].clamp(min=0).max(dim=1)[0].unsqueeze(0)
        s = s.clamp(min=0).max(dim=1)[0].unsqueeze(0)
        s = (s - s.min()) / (s.max() - s.min())
        attn_heads = (attn_heads - attn_heads.min()) / (attn_heads.max() - attn_heads.min())
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        fused = avg_heads * s
        all_layer_attentions.append(fused)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer)
    p = rollout[0 , cls_pos , :].unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits
 

def generate_rollout(model, image, start_layer=15, num_layers=24):
    image.requires_grad_()
    logits = model(image)    
    all_layer_attentions = []
    cls_pos =1024//2
    for layer in range(num_layers):
        attn_heads = model.layers[layer].mixer.xai_b
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    p = rollout[0 , cls_pos , :].unsqueeze(0)
    p = torch.cat([p[:,:cls_pos], p[:,(cls_pos+1):]], dim=-1)
    return p.clamp(min=0).squeeze().unsqueeze(0), logits    