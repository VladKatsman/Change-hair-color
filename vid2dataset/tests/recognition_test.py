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
        self.features = vgg_pretrained_features
        self.verifier = torch.nn.Sequential()
        for x in range(2):
            self.verifier.add_module(str(x), model.classifier[x])
        for x in range(3, 5):
            self.verifier.add_module(str(x), model.classifier[x])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.verifier(x)
        return x


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
    feats = None
    with torch.no_grad():
        pbar = tqdm(test_loader, unit='batches')
        for i, (input, landmarks, bboxes, target) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            feats = torch.cat((feats, output)) if feats is not None else output

    index = 0
    curr_feat = feats[index]
    for i in range(index + 1, feats.shape[0]):
        target_feat = feats[i]
        # dist = torch.abs(curr_feat - target_feat).mean()
        dist = torch.norm(curr_feat - target_feat, 2)
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
