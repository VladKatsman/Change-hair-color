import os
import random
import time
import functools
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils as tutils
import torchvision.transforms as transforms
import torchvision.models as models

from tensorboardX import SummaryWriter
from tqdm import tqdm
from face_tran.utils.obj_factory import obj_factory
from face_tran.utils import utils
from face_tran.utils import seg_utils

from face_tran.utils import losses
from face_tran.utils import seg_transforms


def main(exp_dir='/data/experiments', pretrained_dir=None, train_dir=None, val_dir=None, workers=4, iterations=None, epochs=90,
         start_epoch=0, batch_size=1, resume_dir=None, seed=None, gpus=None, tensorboard=False,
         train_dataset=None, val_dataset=None,
         optimizer='optim.SGD(lr=0.1,momentum=0.9,weight_decay=1e-4)',
         scheduler='lr_scheduler.StepLR(step_size=30,gamma=0.1)',
         log_freq=20, pil_transforms=None, tensor_transforms=None, visual_augmentations=None, geo_augmentations=None,
         arch='resnet18', pretrained=False, cudnn_benchmark=True):
    best_iou = 0

    # Validation
    if not os.path.isdir(exp_dir):
        raise RuntimeError('Experiment directory was not found: \'' + exp_dir + '\'')

    # Seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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

    # Initialize transforms
    pil_transforms = [obj_factory(t) for t in pil_transforms] if type(pil_transforms) == list else pil_transforms
    pil_transforms = seg_transforms.ComposePair(pil_transforms)
    visual_augmentations = [obj_factory(t) for t in visual_augmentations] if visual_augmentations else []
    if visual_augmentations:
        visual_augmentations = transforms.Compose(visual_augmentations)
    # geo_augmentations = [obj_factory(t) for t in geo_augmentations] if visual_augmentations else []
    # if geo_augmentations:
    #     geo_augmentations = transforms.Compose(geo_augmentations)
    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    if not tensor_transforms:
        tensor_transforms = [transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    tensor_transforms = transforms.Compose(tensor_transforms)

    # Initialize datasets
    val_dataset = train_dataset if val_dataset is None else val_dataset
    train_dataset = obj_factory(train_dataset, train_dir, pil_transforms=pil_transforms,
                                tensor_transforms=tensor_transforms, visual_augmentations=visual_augmentations,
                                geo_augmentations=geo_augmentations)

    if val_dir:
        val_dataset = obj_factory(val_dataset, val_dir, pil_transforms=pil_transforms,
                                  tensor_transforms=tensor_transforms)
    # Initialize data loaders
    if iterations is None:
        train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, len(train_dataset),
                                                                  replacement=False)
    else:
        train_sampler = tutils.data.sampler.WeightedRandomSampler(train_dataset.weights, iterations,
                                                                  replacement=False)
    train_loader = tutils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    if val_dir:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=workers)

    # Create model
    model = obj_factory(arch)

    if not pretrained:
        model.apply(utils.init_weights)
    model.to(device)

    # Optimizer and scheduler
    optimizer = obj_factory(optimizer, model.parameters())
    scheduler = obj_factory(scheduler, optimizer)

    # optionally resume from a checkpoint
    checkpoint_dir = pretrained_dir
    model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
        else:
            print("=> randomly initializing model...")

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # define running metrics
    running_metrics_val = seg_utils.Metrics(train_dataset.n_classes)

    # if input shape are the same for the dataset then set to True, otherwise False
    cudnn.benchmark = cudnn_benchmark

    # training/validation procedure loop
    for epoch in range(start_epoch, epochs):
        if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, logger, log_freq)

        # evaluate on validation set
        val_loss, score, report = validate(val_loader, model, criterion, epoch, epochs, device, logger, running_metrics_val)

        # remember best MeanIoU and save checkpoint
        iou = score["Mean IoU : \t"]
        is_best = iou > best_iou
        best_iou = max(iou, best_iou)
        utils.save_checkpoint(exp_dir, 'model', {
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.module.state_dict() if gpus and len(gpus) > 1 else model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'report': report,
        }, is_best)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)


def train(train_loader, model, criterion, optimizer, epoch, device, logger, log_freq):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    total_iter = len(train_loader) * train_loader.batch_size * epoch
    iters = len(train_loader) * train_loader.batch_size * (epoch + 1)

    # switch to train mode
    model.train()

    # set time
    end = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs.size(0)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        optimizer.zero_grad()
        output = model(inputs)

        # compute gradient and do OPTIMIZER step
        loss_sum = 0

        for n in range(len(output)):
            loss_sum += criterion(input=output[n], target=targets)

        loss_sum.backward()
        optimizer.step()

        # update metrics
        losses.update(loss_sum.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_iter += train_loader.batch_size

        # Batch logs (reworked to semseg style)
        if logger and i % log_freq == 0:
            fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
            print_str = fmt_str.format(
                                       total_iter,
                                       iters,
                                       losses.avg,
                                       batch_time.avg / batch_size
            )
            print(print_str)
            logger.add_scalars('iterations', {'loss': losses.val}, total_iter)

    # Epoch logs
    if logger:
        logger.add_scalars('epoch_loss', {'train_loss': losses.avg}, epoch)


def validate(val_loader, model, criterion, epoch, epochs, device, logger, running_metrics_val):
    batch_time = utils.AverageMeter()
    val_los = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, unit='batches')
        for i, (inputs, targets, images) in enumerate(pbar):

            inputs = inputs.to(device)
            targets = targets.to(device)
            images = images.to(device)

            # compute output and loss
            output = model(inputs)
            loss_sum = 0
            for n in range(len(output)):
                loss_sum += criterion(input=output[n], target=targets)

            # update metrics
            pred = torch.zeros_like(output[0])
            for n in range(len(output)):
                pred += output[n].data
            pred = pred / len(output)
            pred = pred.max(1)[1]
            gt = targets.data

            running_metrics_val.update(gt, pred)
            val_los.update(loss_sum.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_description(
                'VALIDATION: Epoch: {} / {}; '
                'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '
                'Loss {loss.val:.3f} ({loss.avg:.3f}); '.format(
                    epoch + 1, epochs, batch_time=batch_time,
                    loss=val_los))

    # Metrics
    score, class_iou = running_metrics_val.get_scores()
    running_metrics_val.reset()

    # Epoch logs
    if logger:
        report = []
        for k, v in score.items():
            k, v = k, "{:.3f}".format(v.item())
            report.append((k, v))
            logger.add_scalar('val_metrics/{}'.format(k.strip()), float(v), epoch)
            print(k, v)
        print("val_loss: {:.3f}".format(val_los.avg))
        report.append("val_loss: {:.3f}".format(val_los.avg))
        for k, v in class_iou.items():
            k, v = k, "{:.3f}".format(v.item())
            report.append((k, v))
            logger.add_scalar('class_metrics/cls_{}'.format(k), float(v), epoch)
        logger.add_scalars('epoch_loss', {'val_loss': val_los.avg}, epoch)

        # Log images
        blend = seg_utils.blend_seg_label(images, pred)
        gt_blend = seg_utils.blend_seg_label(images, targets)
        pred = pred.view(pred.shape[0], 1, pred.shape[1], pred.shape[2]).repeat(1, 3, 1, 1).float()
        grid = seg_utils.make_grid(images, gt_blend, pred, blend,  min(8, val_loader.batch_size))
        logger.add_image('%d/rec' % (256), grid, epoch)

    return val_los.avg, score, report


if __name__ == "__main__":
    #Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('Classifier Training')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('-pre', '--pretrained_dir',  type=str,
                        help='path to directory with pretrained model')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val', default=None, type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, type=int, metavar='N',
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
    parser.add_argument('-va', '--visual_augmentations', default=None, type=str, nargs='+',
                        help='Visual augmentations to input images only')
    parser.add_argument('-ga', '--geo_augmentations', default=None, type=str, nargs='+',
                        help='Geometrical augmentations both to images and ground truth')
    parser.add_argument('-tt', '--tensor_transforms', default=None, type=str, nargs='+', help='tensor transforms')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-cb', '--cudnn_benchmark', default=True, action='store_true',
                        help='if input shapes are the same for the dataset then set to True, otherwise False')
    args = parser.parse_args()
    main(args.exp_dir, args.pretrained_dir, args.train, args.val, workers=args.workers, iterations=args.iterations,
         epochs=args.epochs, start_epoch=args.start_epoch, batch_size=args.batch_size, resume_dir=args.resume,
         seed=args.seed, gpus=args.gpus, tensorboard=args.tensorboard, optimizer=args.optimizer,
         scheduler=args.scheduler, log_freq=args.log_freq, train_dataset=args.train_dataset,
         val_dataset=args.val_dataset, pil_transforms=args.pil_transforms,
         visual_augmentations=args.visual_augmentations, geo_augmentations=args.geo_augmentations,
         tensor_transforms=args.tensor_transforms, arch=args.arch, pretrained=args.pretrained,
         cudnn_benchmark=args.cudnn_benchmark)
