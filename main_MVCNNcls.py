#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
import argparse
import builtins
import os
import random
import shutil
import time
import socket
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import moco.loader
from server_model import serverModel
from custom_dataset import MultiViewDataSet

class SplitImageTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        q = self.transform1(x)
        k = self.transform2(x)
        return [q, k]

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_un', '--unsupervised-learning-rate', default=100.0, type=float,
                    metavar='LR', help='initial learning rate for final linear layer', dest='lr_un')
#parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
#                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default=None, nargs='+', type=int,
                    help='GPU id(s) to use. Default is all visible GPUs.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--num_clients', default=12, type=int,
                    help='number of clients')
parser.add_argument('--mode', default="supervised", type=str,
        help='Mode of VFL to use: supervised, unsupervised, or semi-supervised')
parser.add_argument('--labeled_frac', default=1.0, type=float,
                    help='fraction of training data that is labeled')

args = parser.parse_args()

def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    save_folder_terms = [
        f'b{args.batch_size}',
        f'lr{args.lr:g}',
        f'mode{args.mode}',
        f'frac{args.labeled_frac}',
        f'e{",".join(map(str, args.schedule))},{args.epochs}',
        f'seed{args.seed}',
    ]

    args.save_folder = os.path.join(
        os.path.split(args.pretrained)[0],
        'MVCNNcls',
        os.path.basename(args.pretrained),
        '_'.join(save_folder_terms),
    )
    os.makedirs(args.save_folder, exist_ok=True)
    print(f"save_folder: '{args.save_folder}'")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))

    if args.multiprocessing_distributed and len(args.gpus) == 1:
        warnings.warn('You have chosen to use multiprocessing distributed '
                      'training. But only one GPU is available on this node. '
                      'The training will start within the launching process '
                      'instead to minimize process start overhead.')
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Assuming we have len(args.gpus) processes per node, we need to adjust
        # the total world_size accordingly
        args.world_size = len(args.gpus) * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(index, args):
    # We will do a bunch of `setattr`s such that
    #
    # args.rank               the global rank of this process in distributed training
    # args.index              the process index to this node
    # args.gpus               the GPU ids for this node
    # args.gpu                the default GPU id for this node
    # args.batch_size         the batch size for this process
    # args.workers            the data loader workers for this process
    # args.seed               if not None, the seed for this specific process, computed as `args.seed + args.rank`

    args.index = index
    args.gpu = args.gpus[index]
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    # suppress printing for all but one device per node
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Use GPU(s): {args.gpus} for training on '{socket.gethostname()}'")

    # init distributed training if needed
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            ngpus_per_node = len(args.gpus)
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + index
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size and data
            # loader workers based on the total number of GPUs we have.
            assert args.batch_size % ngpus_per_node == 0
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    if args.seed is not None:
        args.seed = args.seed + args.rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = True

    # build data loaders before initializing model, since we need num_classes for the latter
    train_loader, val_loader, classes = create_data_loaders(args)

    # Create models
    models = []
    optimizers = []
    for m in range(args.num_clients+1):
        # create model
        if m != args.num_clients:
            print(f"=> creating model '{args.arch}' with {len(classes)} classes")
            model = torchvision.models.__dict__[args.arch](num_classes=args.moco_dim)

            # load from pre-trained, before DistributedDataParallel constructor
            if args.mode != "supervised":
                # TODO change filename check to be specific for the client
                if os.path.isfile(args.pretrained):
                    filename_arr = args.pretrained.split('_')
                    filename_arr[1] = f"mvcnn{m}"
                    filename = '_'.join(filename_arr)

                    print("=> loading checkpoint '{}'".format(filename))
                    checkpoint = torch.load(filename, map_location="cpu")

                    # rename moco pre-trained keys
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        # retain only encoder_q up to before the embedding layer
                        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                            # remove prefix
                            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]

                    args.start_epoch = 0
                    msg = model.load_state_dict(state_dict, strict=False)
                    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                    print("=> loaded pre-trained model '{}'".format(args.pretrained))
                else:
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrained))

            # freeze all layers but the last fc
            #if args.mode == "unsupervised":
            #    for name, param in model.named_parameters():
            #        param.requires_grad = False
        else:
            print(f"=> creating server model")
            model = serverModel(num_clients=args.num_clients, num_classes=len(classes), dim=args.moco_dim)
            print("Number of classes:",len(classes))
            # init the fc layer
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

        model.cuda(args.gpu)
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.multiprocessing_distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpus)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=args.gpus)
            else:
                model = torch.nn.DataParallel(model, device_ids=args.gpus)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        if args.mode == "unsupervised" and m != args.num_clients:
            # optimize only the linear classifier
            parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
            #assert len(parameters) == 2  # fc.weight, fc.bias
        else:
            parameters = model.parameters()
        if m != args.num_clients:
            optimizer = torch.optim.SGD(parameters, args.lr,
                                        momentum=args.momentum,
                                        weight_decay=1e-4)
        else:
            lr = args.lr
            if args.mode == "unsupervised":
                lr = args.lr_un
            optimizer = torch.optim.SGD(parameters, lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        models.append(model)
        optimizers.append(optimizer)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if isinstance(best_acc1, torch.Tensor):
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_acc1 = 0

    if args.evaluate:
        validate(val_loader, models, criterion, args)
        return

    train_loss = []
    train_acc1 = []
    train_acc5 = []
    test_loss = []
    test_acc1 = []
    test_acc5 = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for m in range(args.num_clients+1):
            lr = args.lr
            if m == args.num_clients and args.mode == "unsupervised":
                lr = args.lr_un
            adjust_learning_rate(optimizers[m], epoch, lr)

        # train for one epoch
        loss, acc1, acc5 = train(train_loader, models, criterion, optimizers, epoch, args)
        train_loss.append(loss)
        train_acc1.append(acc1)
        train_acc5.append(acc5)

        # evaluate on validation set
        loss, acc1, acc5 = validate(val_loader, models, criterion, args)
        test_loss.append(loss)
        test_acc1.append(acc1)
        test_acc5.append(acc5)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            print(f"New best Acc1 {best_acc1:.4f}")

        if (args.distributed and args.rank == 0) or (args.index == 0):
            pickle.dump(train_loss, open(os.path.join(args.save_folder,'train_loss.pkl'), 'wb'))
            pickle.dump(train_acc1, open(os.path.join(args.save_folder,'train_acc1.pkl'), 'wb'))
            pickle.dump(train_acc5, open(os.path.join(args.save_folder,'train_acc5.pkl'), 'wb'))
            pickle.dump(test_loss, open(os.path.join(args.save_folder,'test_loss.pkl'), 'wb'))
            pickle.dump(test_acc1, open(os.path.join(args.save_folder,'test_acc1.pkl'), 'wb'))
            pickle.dump(test_acc5, open(os.path.join(args.save_folder,'test_acc5.pkl'), 'wb'))

            for m in range(args.num_clients+1):
                save_filename = os.path.join(args.save_folder, f'checkpoint_client{m}.pth.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': models[m].state_dict(),
                    'best_acc1': best_acc1,
                    'acc1': acc1,
                    'acc5': acc5,
                    'optimizer' : optimizers[m].state_dict(),
                }, is_best, save_filename)
                print(f"saved to '{save_filename}'")


def create_data_loaders(args):
    # Data loading code
    traindir = args.data
    valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = MultiViewDataSet(traindir, 'train',
            transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]))

    indices = torch.randperm(len(train_dataset))
    train_dataset_sub = torch.utils.data.Subset(train_dataset, indices[:int(len(train_dataset)*args.labeled_frac)])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_sub, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset_sub, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        MultiViewDataSet(traindir, 'test',
                transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes 


def train(train_loader, models, criterion, optimizers, epoch, args):
    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')
    losses = utils.AverageMeter('Loss', '.4e')
    top1 = utils.AverageMeter('Acc1', '6.2f')
    top5 = utils.AverageMeter('Acc5', '6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, utils.ProgressMeter.BR, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    for client in range(args.num_clients+1):
        if args.mode == "unsupervised" and client != args.num_clients: 
            models[client].eval()
        else:
            models[client].train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        for client in range(args.num_clients):
            images[client] = images[client].cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        embeddings = []
        for client in range(args.num_clients):
            image_local = images[client]
            embeddings.append(models[client](image_local))
        output = models[-1](torch.cat(embeddings,axis=1))
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss, images[0].size(0))
        top1.update(acc1, images[0].size(0))
        top5.update(acc5, images[0].size(0))

        # compute gradient and do SGD step
        for client in range(args.num_clients+1):
            optimizers[client].zero_grad()
        loss.backward()
        for client in range(args.num_clients+1):
            if args.mode != "unsupervised" or client == args.num_clients: 
                optimizers[client].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    loss, acc1, acc5 = losses.avg, top1.avg, top5.avg
    print(f'Training * Loss {loss:.5f} Acc1 {acc1:.3f} Acc5 {acc5:.3f}')
    return loss, acc1, acc5


def validate(val_loader, models, criterion, args):
    batch_time = utils.AverageMeter('Time', '6.3f')
    losses = utils.AverageMeter('Loss', '.4e')
    top1 = utils.AverageMeter('Acc1', '6.2f')
    top5 = utils.AverageMeter('Acc5', '6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, utils.ProgressMeter.BR, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    for i in range(args.num_clients+1):
        models[i].eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            for client in range(args.num_clients):
                images[client] = images[client].cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            embeddings = []
            for client in range(args.num_clients):
                image_local = images[client]
                embeddings.append(models[client](image_local))
            output = models[-1](torch.cat(embeddings,axis=1))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss, images[0].size(0))
            top1.update(acc1, images[0].size(0))
            top5.update(acc5, images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    loss, acc1, acc5 = losses.avg, top1.avg, top5.avg
    print(f'Test * Loss {loss:.5f} Acc1 {acc1:.3f} Acc5 {acc5:.3f}')

    return loss, acc1, acc5


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights):
    r"""
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def adjust_learning_rate(optimizer, epoch, lr):
    """Decay the learning rate based on schedule"""
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
