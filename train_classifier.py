import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils as tutils
import torchvision.transforms as transforms
import torchvision.models as models
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from face_tran.utils.obj_factory import obj_factory
from face_tran.utils import utils


def main(exp_dir='/data/experiments', pretrained_model=None, train_dir=None, val_dir=None, workers=4, iterations=None, epochs=180, start_epoch=0,
         batch_size=64, learning_rate=0.1, momentum=0.9, weight_decay=1e-4, resume_dir=None, seed=None,
         gpus=None, tensorboard=False,
         train_dataset=None, val_dataset=None,
         optimizer='optim.SGD(lr=0.1,momentum=0.9,weight_decay=1e-4)',
         scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
         log_freq=20, pil_transforms=None, tensor_transforms=None, arch='resnet18'):
    best_prec1 = 0

    # Validation
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')


    # Check CUDA device availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    # Initialize loggers
    logger = SummaryWriter(log_dir=exp_dir) if tensorboard else None

    # Initialize datasets
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    if not tensor_transforms:
        tensor_transforms = [transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    img_transforms = transforms.Compose(pil_transforms + tensor_transforms)

    val_dataset = train_dataset if val_dataset is None else val_dataset
    train_dataset = obj_factory(train_dataset, train_dir, transform=img_transforms)
    if val_dir:
        val_dataset = obj_factory(val_dataset, val_dir, transform=img_transforms)

    # Initialize data loaders
    sampler = None
    if iterations is not None:
        sampler = tutils.data.sampler.SubsetRandomSampler(np.random.choice(len(train_dataset), iterations))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True, shuffle=not sampler)
    if val_dir:
        if iterations is not None:
            sampler = tutils.data.sampler.SubsetRandomSampler(np.random.choice(len(val_dataset), iterations))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=workers, pin_memory=True, drop_last=True, shuffle=not sampler)

    # Create model
    model = obj_factory(arch)
    model.apply(utils.init_weights)
    if pretrained_model is not None:
        pretrained_weights = torch.load(pretrained_model)
        if arch.startswith('vgg'):
            model.features.load_state_dict(pretrained_weights, strict=False)
        else:
            model.load_state_dict(pretrained_weights, strict=False)

    # Optimizer and scheduler
    optimizer = obj_factory(optimizer, model.parameters())
    scheduler = obj_factory(scheduler, optimizer)

    # Optionally resume from a checkpoint
    checkpoint_dir = exp_dir if resume_dir is None else resume_dir
    model_path = os.path.join(checkpoint_dir, 'model_latest.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))


    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus).to(device)
    else:
        model.to(device)


    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    for epoch in range(start_epoch, epochs):
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, epochs, device, logger, log_freq)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion, epoch, epochs, device, logger, log_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint(exp_dir, 'model', {
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.module.state_dict() if gpus and len(gpus) > 1 else model.state_dict(),
            'best_prec1': prec1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)


def train(train_loader, model, criterion, optimizer, epoch, epochs, device, logger, log_freq):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    total_iter = len(train_loader) * train_loader.batch_size * epoch

    # switch to train mode
    model.train()

    end = time.time()
    pbar = tqdm(train_loader, unit='batches')
    for i, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        if len(output.shape) > 2:
            output = output.view(output.size(0), -1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_iter += train_loader.batch_size

        # Batch logs
        pbar.set_description(
            'TRAINING: Epoch: {} / {}; '
            'Timing: [Data: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
            'Batch: {data_time.val:.3f} ({data_time.avg:.3f})]; '
            'Loss {loss.val:.4f} ({loss.avg:.4f}); '
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}); '
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, epochs, batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1, top5=top5))

        if logger and i % log_freq == 0:
            logger.add_scalars('batch', {'loss': losses.val}, total_iter)
            logger.add_scalars('batch/acc', {'top1': top1.val, 'top5': top5.val}, total_iter)

    # Epoch logs
    if logger:
        logger.add_scalars('epoch/acc/train', {'top1': top1.avg, 'top5': top5.avg}, epoch)


def validate(val_loader, model, criterion, epoch, epochs, device, logger, log_freq):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, unit='batches')
        for i, (input, target) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            if len(output.shape) > 2:
                output = output.view(output.size(0), -1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(
                'VALIDATION: Epoch: {} / {}; '
                'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '
                'Loss {loss.val:.3f} ({loss.avg:.3f}); '
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}); '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, epochs, batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

        # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

        # Epoch logs
        if logger:
            logger.add_scalars('epoch/acc/val', {'top1': top1.avg, 'top5': top5.avg}, epoch)

    return losses.avg, top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    #Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('Classifier Training')
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('-pre', '--pretrained_model', default=None,
                        help='path to the directory with pretrained model')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val', default=None, type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-se', '--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-sr', '--start_res', default=4, type=int,
                        metavar='N', help='starting training resolution (must be power of 2)')
    parser.add_argument('-m', '--max_res', default=256, type=int,
                        metavar='N', help='maximum training resolution (must be power of 2)')
    parser.add_argument('-fc', '--feature_channels', default=512, type=int, metavar='N',
                        help='max number of channels of the embedding feature')
    parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                        metavar='F', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='F',
                        help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                        metavar='F', help='weight decay (default: 1e-4)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-td', '--train_dataset', default='generic_face_dataset.GenericFaceDataset', type=str,
                        help='train dataset object')
    parser.add_argument('-vd', '--val_dataset', default=None, type=str, help='val dataset object')
    parser.add_argument('-o', '--optimizer', default='optim.SGD(lr=0.1,momentum=0.9,weight_decay=1e-4)', type=str,
                        help='optimizer object')
    parser.add_argument('--scheduler', default='lr_scheduler.StepLR(step_size=30,gamma=0.1)', type=str,
                        help='scheduler object')
    parser.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                        help='number of steps between each loss plot')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', default=None, type=str, nargs='+', help='tensor transforms')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg',
                        help='model architecture')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    args = parser.parse_args()
    main(args.exp_dir, args.pretrained_model, args.train, args.val, workers=args.workers, iterations=args.iterations,
         epochs=args.epochs, start_epoch=args.start_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate,
         momentum=args.momentum, weight_decay=args.weight_decay, resume_dir=args.resume, seed=args.seed,
         gpus=args.gpus, tensorboard=args.tensorboard, optimizer=args.optimizer, scheduler=args.scheduler,
         log_freq=args.log_freq, train_dataset=args.train_dataset, val_dataset=args.val_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms, arch=args.arch)
