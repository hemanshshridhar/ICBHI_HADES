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
from models import get_backbone_class

from models_DASS.ast_models_vs import DASS


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

    # optimization
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
                        help='weighted cross entropy loss')
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true')
    parser.add_argument('--domain_adaptation2', action='store_true')
    parser.add_argument('--domain_adaptation', action='store_true')

    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/icbhi_dataset')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)

    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound')
    parser.add_argument('--n_cls', type=int, default=4)
    parser.add_argument('--test_fold', type=str, default='official',
                        choices=['official', '0', '1', '2', '3', '4'])
    parser.add_argument('--weighted_sampler', action='store_true')
    parser.add_argument('--stetho_id', type=int, default=-1)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--butterworth_filter', type=int, default=None)
    parser.add_argument('--desired_length', type=int, default=8)
    parser.add_argument('--nfft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--concat_aug_scale', type=float, default=0)
    parser.add_argument('--pad_types', type=str, default='repeat')
    parser.add_argument('--resz', type=float, default=1)
    parser.add_argument('--raw_augment', type=int, default=0)
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup')
    parser.add_argument('--specaug_mask', type=str, default='mean',
                        choices=['mean', 'zero'])

    # model
    parser.add_argument('--model', type=str, default='dass')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--from_sl_official', action='store_true')
    parser.add_argument('--ma_update', action='store_true')
    parser.add_argument('--ma_beta', type=float, default=0)
    parser.add_argument('--audioset_pretrained', action='store_true')

    # method
    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--negative_pair', type=str, default='all',
                        choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='project1_project2block')
    parser.add_argument('--meta_mode', type=str, default='dev',
                        choices=['none', 'age', 'sex', 'loc', 'dev', 'label'])

    # HADES params
    parser.add_argument('--hades_on', action='store_true', default=False,
                        help='enable HADES blocks inside VMamba')
    parser.add_argument('--hades_blocks', type=str, default='2,3,12',
                        help='comma separated block indices within hades_stage')
    parser.add_argument('--hades_stage', type=str, default='2',
                        help='comma separated stage indices where HADES is active')
    parser.add_argument('--alpha_lb', type=float, default=1.0,
                        help='weight for load balance loss from HADES')
    parser.add_argument('--alpha_div', type=float, default=1.0,
                        help='weight for diversity loss from HADES')

    args = parser.parse_args()

    # parse lr decay epochs
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = []
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # parse hades_blocks and hades_stage into sets
    args.hades_blocks = set([int(x) for x in args.hades_blocks.split(',') if x != ''])
    args.hades_stage  = set([int(x) for x in args.hades_stage.split(',')  if x != ''])

    # model name
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method) \
        if args.meta_mode == 'none' \
        else '{}_{}_{}_{}'.format(args.dataset, args.model, args.method, args.meta_mode)
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
            if args.domain_adaptation or args.domain_adaptation2:
                if args.meta_mode == 'age':
                    args.meta_cls_list = ['Adult', 'Child']
                elif args.meta_mode == 'sex':
                    args.meta_cls_list = ['Male', 'Female']
                elif args.meta_mode == 'loc':
                    args.meta_cls_list = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
                elif args.meta_mode == 'dev':
                    args.meta_cls_list = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L']
                elif args.meta_mode == 'label':
                    args.meta_cls_list = ['None', 'Crackle', 'Wheeze', 'Both']

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

    if args.n_cls == 0 and args.m_cls != 0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args


def set_loader(args):
    if args.dataset == 'icbhi':
        args.h, args.w = 1024, 128
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            SpecAugment(args),
            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))
        ])

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform,
                                     args=args, print_flag=True)
        val_dataset   = ICBHIDataset(train_flag=False, transform=val_transform,
                                     args=args, print_flag=True)
        args.class_nums = train_dataset.class_nums
    else:
        raise NotImplementedError

    if args.weighted_sampler:
        reciprocal_weights = []
        for idx in range(len(train_dataset)):
            reciprocal_weights.append(train_dataset.class_ratio[train_dataset.labels[idx]])
        weights = (1 / torch.Tensor(reciprocal_weights))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=True, sampler=sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, sampler=None)

    return train_loader, val_loader, args


def set_model(args):
    if args.model == 'dass':
        model = DASS(
            label_dim=args.n_cls,
            imagenet_pretrain=args.from_sl_official,
            audioset_pretrain=args.audioset_pretrained,
            model_size='medium',
            hades_on=args.hades_on,
            hades_blocks=args.hades_blocks,  # already a set from parse_args
            hades_stage=args.hades_stage,    # already a set from parse_args
        )
    else:
        kwargs = {}
        model = get_backbone_class(args.model)(**kwargs)

    if not args.weighted_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights)

    criterion = criterion.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.cuda()
    optimizer = set_optimizer(args, list(model.parameters()))

    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    model.train()

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        if args.ma_update:
            with torch.no_grad():
                ma_ckpt = [deepcopy(model.state_dict())]
                p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

        data_time.update(time.time() - end)
        images = images.squeeze().cuda(non_blocking=True)
        class_labels = labels.cuda(non_blocking=True)
        bsz = class_labels.shape[0]

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                # DASS returns (logits, lb_loss, div_loss)
                logits, lb_loss, div_loss = model(images)
                class_loss = criterion(logits, class_labels)
                loss = class_loss

                # add HADES auxiliary losses if enabled
                if args.hades_on:
                    loss = loss + args.alpha_lb * lb_loss + args.alpha_div * div_loss

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(logits[:bsz], class_labels, topk=(1,))
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
                   epoch, idx + 1, len(train_loader),
                   batch_time=batch_time, data_time=data_time,
                   loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()

    batch_time = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    hits   = [0.0] * args.n_cls
    counts = [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.squeeze().cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.cuda.amp.autocast():
                # unpack — lb_loss and div_loss not needed at eval
                logits, _, _ = model(images)
                loss = criterion(logits, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(logits, 1)
            for i in range(preds.shape[0]):
                counts[labels[i].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[i].item() == labels[i].item():
                        hits[labels[i].item()] += 1.0
                else:
                    if labels[i].item() == 0 and preds[i].item() == labels[i].item():
                        hits[labels[i].item()] += 1.0
                    elif labels[i].item() != 0 and preds[i].item() > 0:
                        hits[labels[i].item()] += 1.0

            sp, se, sc = get_score(hits, counts)
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader),
                       batch_time=batch_time, loss=losses, top1=top1))

    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} '
          '(Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(
          sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool


def main():
    args = parse_args()

    # save args — convert sets to lists for JSON serialization
    args_dict = vars(args).copy()
    if isinstance(args_dict.get('hades_blocks'), set):
        args_dict['hades_blocks'] = list(args_dict['hades_blocks'])
    if isinstance(args_dict.get('hades_stage'), set):
        args_dict['hades_stage'] = list(args_dict['hades_stage'])
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

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
    model, criterion, optimizer = set_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    print('*' * 20)
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs + 1):
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            loss, acc = train(train_loader, model, criterion, optimizer, epoch, args)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            best_acc, best_model, save_bool = validate(
                val_loader, model, criterion, args, best_acc, best_model)

            if save_bool:
                save_file = os.path.join(
                    args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(
                    best_acc[2], epoch))
                save_model(model, optimizer, args, epoch, save_file)

            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file)

        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        save_model(model, optimizer, args, epoch, save_file)

    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _ = validate(val_loader, model, criterion, args, best_acc)

    update_json('%s' % args.model_name, best_acc,
                path=os.path.join(args.save_dir, 'results.json'))


if __name__ == '__main__':
    main()