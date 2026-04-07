from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_dataset import ICBHIDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import PatchMixLoss, PatchMixConLoss

from models_DASS.ast_models_v2 import DASS
def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save/')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')

    # optimizationc
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lr_decay_epochs', type=str, default='35,40')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-7)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')
    parser.add_argument('--domain_adaptation2', action='store_true')
    parser.add_argument('--domain_adaptation', action='store_true')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/icbhi_dataset')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=4,
                        help='set k-way classification problem')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--weighted_sampler', action='store_true',
                        help='weighted sampler inversly proportional to class ratio')
    parser.add_argument('--stetho_id', type=int, default=-1,
                        help='stethoscope device id, use only when finetuning on each stethoscope data')
    parser.add_argument('--sample_rate', type=int,  default=16000,
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--butterworth_filter', type=int, default=None,
                        help='apply specific order butterworth band-pass filter')
    parser.add_argument('--desired_length', type=int,  default=8,
                        help='fixed length size of individual cycle')
    parser.add_argument('--nfft', type=int, default=1024,
                        help='the frequency size of fast fourier transform')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--concat_aug_scale', type=float,  default=0,
                        help='to control the number (scale) of concatenation-based augmented samples')
    parser.add_argument('--pad_types', type=str,  default='repeat',
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1,
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0,
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup',
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean',
                        help='specaug mask value', choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='dass')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true', help='load from imagenet- and audioset-pretrained model')

    parser.add_argument('--method', type=str, default='ce')
    # Patch-Mix CL loss
    parser.add_argument('--negative_pair', type=str, default='all', help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.20)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--target_type', type=str, default='grad_block', help='how to make target representation',
                        choices=['project_flow_all', 'representation_all', 'z1block_project', 'z1_project2', 'project1block_project2', 'project1_r2block', 'project1_r2', 'project1_project2block', 'project_block_all', 'grad_block', 'grad_flow', 'project_block', 'project_flow'])
    parser.add_argument('--time_only', action='store_true')
    parser.add_argument('--cl_mode', type=str, default='dual', choices=['time', 'freq', 'dual', '2d'])
    parser.add_argument('--alpha_time', type=float, default=1.0)
    parser.add_argument('--alpha_freq', type=float, default=1.0)

    
    # Lung-SRAD
    parser.add_argument('--gaussian_blur', action='store_true', help='enable gaussian blur inside VMamba blocks')
    parser.add_argument('--kernel_size', type=int, default=5,help='gaussian blur kernel size')
    parser.add_argument('--sigma', type=float, default=3.0, help='gaussian blur sigma')
    parser.add_argument('--blur_blocks', type=str, default='2,3', help='comma separated block indices where blur is applied')
    parser.add_argument('--blur_stage', type=str, default='2', help='comma separated stage indices where blur is active')    
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    args.blur_blocks = set([int(x) for x in args.blur_blocks.split(',') if x != ''])
    args.blur_stage = set([int(x) for x in args.blur_stage.split(',') if x != ''])      
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)


    args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.m_cls = 4


    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']

        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError

    else:
        raise NotImplementedError

    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args


def set_loader(args):
    if args.dataset == 'icbhi':


        args.h, args.w = 1024, 128
        train_transform = [transforms.ToTensor(),
                            SpecAugment(args),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]


        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)

        args.class_nums = train_dataset.class_nums
    else:
        raise NotImplemented

    if args.weighted_sampler:
        reciprocal_weights = []
        for idx in range(len(train_dataset)):
            reciprocal_weights.append(train_dataset.class_ratio[train_dataset.labels[idx]])
        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=sampler is None,
                                               num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, sampler=None)

    return train_loader, val_loader, args


def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  

    if args.model == 'dass':
        model = DASS(label_dim=4, imagenet_pretrain=False, audioset_pretrain=args.audioset_pretrained, 
                     enable_patch_mix=True if args.method in ['patchmix', 'patchmix_cl'] else False, mix_beta=args.mix_beta, model_size='medium',
                     blur_blocks=args.blur_blocks, gaussian_blur=args.gaussian_blur, kernel_size=args.kernel_size, sigma=args.sigma, blur_stage = args.blur_stage)
    else:
        model = get_backbone_class(args.model)(**kwargs)
    if not args.weighted_loss:
        weights = None
        criterion = nn.CrossEntropyLoss()

    else:
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()

        criterion = nn.CrossEntropyLoss(weight=weights)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method == 'patchmix_cl' else nn.Identity()
    
    if args.method == 'ce':
        criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]

    model.cuda()
    projector.cuda()
    optim_params = list(model.parameters())
    optimizer = set_optimizer(args, optim_params)
    return model, projector, criterion, optimizer


def train(train_loader, model, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    projector.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        if args.ma_update:
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict())]
                p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
        data_time.update(time.time() - end)
        images = images.squeeze()
        images = images.cuda(non_blocking=True)
        class_labels = labels[0].cuda(non_blocking=True)

        bsz = class_labels.shape[0]

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features, output = model(images)
                class_loss = criterion[0](output, class_labels)
                loss = class_loss
            elif args.method == 'patchmix_cl':
                features, output = model(images)
                class_loss = criterion[0](output, class_labels)
                loss = class_loss
                
                if args.target_type == 'grad_block':
                    proj1 = deepcopy(features.detach())
                elif args.target_type == 'grad_flow':
                    proj1 = features
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(projector(features).detach())
                elif args.target_type == 'project_flow':
                    proj1 = projector(features)
                
                if args.cl_mode == 'time':
                    mix_time, labels_a_t, labels_b_t, lam_t, index_t = model(images, y=class_labels, patch_mix=True, mix_type="time")
                    proj_time = projector(mix_time)
                    loss_time = criterion[1](proj1, proj_time, class_labels, labels_b_t, lam_t, index_t, args)
                    loss += args.alpha_time * loss_time
                
                if args.cl_mode == 'freq':
                    mix_freq, labels_a_f, labels_b_f, lam_f, index_f = model(images, y=class_labels, patch_mix=True, mix_type="freq")
                    proj_freq = projector(mix_freq)
                    loss_freq = criterion[1](proj1, proj_freq, class_labels, labels_b_f, lam_f, index_f, args)
                    loss += args.alpha_freq * loss_freq

                elif args.cl_mode == 'dual':
                    # ---- Time CL ----
                    mix_time, labels_a_t, labels_b_t, lam_t, index_t = model(images, y=class_labels, patch_mix=True, mix_type="time")
                    proj_time = projector(mix_time)
                    loss_time = criterion[1](proj1, proj_time, class_labels, labels_b_t, lam_t, index_t, args)
                    loss += args.alpha_time * loss_time

                    # ---- Freq CL ----
                    mix_freq, labels_a_f, labels_b_f, lam_f, index_f = model(images, y=class_labels, patch_mix=True, mix_type="freq")
                    proj_freq = projector(mix_freq)
                    loss_freq = criterion[1](proj1, proj_freq, class_labels, labels_b_f, lam_f, index_f, args)
                    loss += args.alpha_freq * loss_freq


                elif args.cl_mode == '2d': #AST-style patch-mix
                    mix_images, labels_a, labels_b, lam, index = model(images, y=class_labels, patch_mix=True, mix_type="2d")
                    proj2 = projector(mix_images)
                    loss += args.alpha * criterion[1](proj1, proj2, class_labels, labels_b, lam, index, args)
                
        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], class_labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
              
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])

        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.squeeze()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
              features, output = model(images)       
              loss = criterion[0](output, labels)
            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)
            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool


def main():
    args = parse_args()
    args_dict = vars(args).copy()

    # convert non-json types
    if isinstance(args_dict.get("blur_blocks"), set):
        args_dict["blur_blocks"] = list(args_dict["blur_blocks"])

    if isinstance(args_dict.get("blur_stage"), set):
        args_dict["blur_stage"] = list(args_dict["blur_stage"])    
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)        
    # with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
    #     json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    
    model, projector, criterion, optimizer = set_model(args)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['model'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1
    print('*' * 20)
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            loss, acc = train(train_loader, model, projector, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))

            # eval for one epoch
            best_acc, best_model, save_bool = validate(val_loader, model ,criterion, args, best_acc, best_model)

            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:
                save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                save_model(model, optimizer, args, epoch, save_file)

            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file)

        # save a checkpoint of classifier with the best accuracy or score
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])


        save_model(model, optimizer, args, epoch, save_file)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _  = validate(val_loader, model, criterion, args, best_acc)

    update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    print("{} done".format(args.tag))

if __name__ == '__main__':
    main()