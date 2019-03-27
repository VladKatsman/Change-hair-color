import os
import random
import time

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.utils.data
import torch.utils.data.distributed

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from tqdm import tqdm
from face_tran.utils.obj_factory import obj_factory
from face_tran.utils import utils
from convcrf import convcrf


def main(exp_dir='/data/experiments', output_dir='/data/experiments', val_dir=None, workers=4, iterations=None,
         batch_size=1, seed=None, gpus=None, tensorboard=False, val_dataset=None,
         pil_transforms=None, tensor_transforms=None,
         arch='resnet18', cudnn_benchmark=True, use_crf=False):


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


    # Initialize datasets
    if pil_transforms is not None:
        pil_transforms = [obj_factory(t) for t in pil_transforms] if type(pil_transforms) == list else pil_transforms

    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    if not tensor_transforms:
        tensor_transforms = [transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    tensor_transforms = transforms.Compose(tensor_transforms)

    print(val_dir, val_dataset)
    val_dataset = obj_factory(val_dataset, val_dir, pil_transforms=pil_transforms,
                              tensor_transforms=tensor_transforms)

    # Initialize data loaders
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=workers)

    # Create model
    model = obj_factory(arch)
    model.to(device)

    # Load weights
    checkpoint_dir = exp_dir
    model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # if input shape are the same for the dataset then set to True, otherwise False
    cudnn.benchmark = cudnn_benchmark

    # evaluate on validation set
    validate(val_loader, model,  device, val_dir, batch_size)


def validate(val_loader, model, device, val_dir, batch_size):
    batch_time = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        pbar = tqdm(val_loader, unit='batches')
        c = 0
        new_list = []
        for j, (inputs, im_names, hs, ws) in enumerate(pbar):

            _, _, h, w = inputs.size()
            val_inputs = inputs.to(device)

            # compute output and loss
            output = model(val_inputs)

            # getting images and predictions as numpy arrays
            pred = output.data.max(1)[1].cpu().numpy()
            hs = hs.cpu().numpy()
            ws = ws.cpu().numpy()
            for i in range(pred.shape[0]):
                mask = Image.fromarray(pred[i].astype(np.uint8))
                imP = mask.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=3)
                reverse_colors = np.array(imP)

                # we want to exclude poor quality predictions, practically found threshold == 7000
                a = np.sum(reverse_colors[reverse_colors == 1])  # face class
                b = np.count_nonzero(reverse_colors == 0)  # hair class

                if len(np.unique(reverse_colors)) == 3 and a+b > (0.15 * h * w):
                    reverse_colors[reverse_colors == 0] = 3
                    reverse_colors[reverse_colors == 2] = 0
                    reverse_colors[reverse_colors == 3] = 2
                    imP = Image.fromarray(reverse_colors, mode='P')
                    imP.putpalette([
                        0, 0, 0,  # index 0 is black (background)
                        0, 255, 0,  # index 1 is green (face)
                        255, 0, 0,  # index 2 is red (hair)
                    ])

                    # resize to match original image
                    h, w = hs[i], ws[i]
                    imP.resize(size=(h, w))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    folder = os.path.join(val_dir, 'masks', im_names[i].split('/')[0])
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    path_save = os.path.join(folder, "{}.png".format(im_names[i].split('/')[-1]))

                    # save data we need
                    imP.save(path_save)
                    new_list.append(im_names[i])
                    c += 1
                    pbar.set_description(
                        'VALIDATION: Iter: {} / {}; '
                        'Timing: [Batch: {batch_time.val:.3f} ({batch_time.avg:.3f})]; '.format(
                            c, len(val_loader)*batch_size, batch_time=batch_time))

            path_t = os.path.join(val_dir, 'masks.txt')
            with open(path_t, 'w') as t:
                for item in new_list:
                    t.write("%s\n" % item)


def get_labels():
    """Load the mapping that associates pascal classes with label colors

        Array values could be changed according to the task and classes

    Returns:
        np.ndarray with dimensions (N, 3)
    """
    return np.asarray(
        [
            [0, 0, 255],  # background
            [0, 255, 0],  # face
            [255, 0, 0],  # hair

        ]
    )


def decode_segmap(label_mask, n_classes, label_colours=get_labels()):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        n_classes (int): number of classes in the dataset
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(1, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r  # / 255.0
    rgb[:, :, 1] = g  # / 255.0
    rgb[:, :, 2] = b  # / 255.0

    return rgb


if __name__ == "__main__":

    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('Inference of segmentation algorithms')
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser.add_argument('exp_dir',
                        help='path to experiment directory')
    parser.add_argument('--output_dir', default=None, type=str, metavar='DIR',
                        help='path to directory to save predicted masks on images')
    parser.add_argument('-t', '--train', type=str, metavar='DIR',
                        help='paths to train dataset root directory')
    parser.add_argument('-v', '--val_dir', default=None, type=str, metavar='DIR',
                        help='paths to valuation dataset root directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-i', '--iterations', default=None, type=int, nargs='+', metavar='N',
                        help='number of iterations per resolution to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='random seed')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-crf', '--enable_crf', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-vd', '--val_dataset', default=None, type=str, help='val dataset object')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', default=None, type=str, nargs='+', help='tensor transforms')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('-cb', '--cudnn_benchmark', default=True, action='store_true',
                        help='if input shapes are the same for the dataset then set to True, otherwise False')
    args = parser.parse_args()
    main(args.exp_dir, val_dir=args.val_dir, workers=args.workers,
         iterations=args.iterations, batch_size=args.batch_size,
         seed=args.seed, gpus=args.gpus, tensorboard=args.tensorboard, use_crf=args.enable_crf,
         val_dataset=args.val_dataset, pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms,
         arch=args.arch, cudnn_benchmark=args.cudnn_benchmark)
