import os
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.utils as tutils
from torch.nn.modules.distance import PairwiseDistance
import face_gan.utils.utils as utils
from face_gan.utils.obj_factory import obj_factory
import face_gan.data.landmark_transforms as landmark_transforms
from face_gan.models.vgg import vgg19


class Vgg19(torch.nn.Module):
    def __init__(self, model_path=None, requires_grad=False):
        super(Vgg19, self).__init__()
        if model_path is None:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        else:
            model = vgg19(pretrained=False)
            checkpoint = torch.load(model_path)
            del checkpoint['state_dict']['classifier.6.weight']
            del checkpoint['state_dict']['classifier.6.bias']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 13):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 26):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 39):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(39, 46):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6]
        return out


class VGGLoss(nn.Module):
    def __init__(self, model_path=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(model_path)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def main(in_dir, workers=4, batch_size=64, gpus=None, test_dataset='face_landmarks_dataset.FaceLandmarksDataset',
         pil_transforms=None,
         tensor_transforms=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'),
         model_path=None):
    # Validation
    if not os.path.isdir(in_dir):
        raise RuntimeError('Input directory was not found: \'' + in_dir + '\'')

    # Create file list if it doesn't exist
    img_list_file = in_dir + '_list.txt'
    if not os.path.exists(img_list_file):
        in_dir_name = os.path.basename(in_dir)
        img_files = [in_dir_name + '/' + os.path.basename(f) for f in glob(in_dir + '/*.jpg')]
        np.savetxt(img_list_file, img_files, fmt='%s')

    # Landmarks and bounding boxes file paths
    landmarks_file = in_dir + '_landmarks.npy'
    bboxes_file = in_dir + '_bboxes.npy'

    # Check CUDA device availability
    device, gpus = utils.set_device(gpus)

    # Initialize datasets
    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    tensor_transforms = [obj_factory(t) for t in tensor_transforms] if tensor_transforms is not None else []
    img_transforms = landmark_transforms.Compose(pil_transforms + tensor_transforms)
    test_dataset = obj_factory(test_dataset, os.path.dirname(in_dir), transform=img_transforms, img_list=img_list_file,
                               landmarks_list=landmarks_file, bboxes_list=bboxes_file)

    # Initialize data loader
    test_loader = tutils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=None,
                                         num_workers=workers, pin_memory=True, drop_last=False, shuffle=False)

    # Create model
    model = Vgg19(model_path).to(device)

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    model.eval()

    # Process data
    feats = [None]*6
    with torch.no_grad():
        pbar = tqdm(test_loader, unit='batches')
        for i, (input, landmarks, bboxes, target) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            for f in range(6):
                feats[f] = torch.cat((feats[f], output[f])) if feats[f] is not None else output[f]

    s = 5
    index = 0
    curr_feat = feats[s][index].view(-1)
    for i in range(index + 1, feats[s].shape[0]):
        target_feat = feats[s][i].view(-1)
        dist = torch.abs(curr_feat - target_feat).mean()
        print(dist)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('produce')
    parser.add_argument('input', metavar='DIR', help='input directory')
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('-td', '--test_dataset', default='face_landmarks_dataset.FaceLandmarksDataset', type=str,
                        help='test dataset object')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('transforms.ToTensor()', 'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])'))
    parser.add_argument('-m', '--model', metavar='PATH', help='model path')
    args = parser.parse_args()
    main(args.input, workers=args.workers, batch_size=args.batch_size, gpus=args.gpus, test_dataset=args.test_dataset,
         pil_transforms=args.pil_transforms, tensor_transforms=args.tensor_transforms, model_path=args.model)
