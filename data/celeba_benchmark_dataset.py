import os
import collections
import torch
import numpy as np

from PIL import Image
from torch.utils import data


class Celeba(data.Dataset):
    """Data loader for the CelebA Aligned benchmark face-hair semantic segmentation dataset.

        Args:
         root (str): path to dataset folder
         pil_transforms (obj): PIL transformations with both input and mask
         tensor_transforms (obj): Tensor transformations with input
    """

    def __init__(self, root, pil_transforms=None, tensor_transforms=None, crf=False):
        self.root = os.path.expanduser(root)
        self.split = "val"
        self.n_classes = 3
        self.files = collections.defaultdict(list)
        self.crf = crf

        # list of images
        path = os.path.join(self.root, self.split + "_img.txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.images = file_list

        # list of segmentations
        path = os.path.join(self.root, self.split + "_seg.txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.segs = file_list
        self.pil_transforms = pil_transforms
        self.tensor_transforms = tensor_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        id = index
        im_name = self.images[id]
        seg_name = self.segs[id]
        im_path = os.path.join(self.root, im_name)
        seg_path = os.path.join(self.root, seg_name)
        image = Image.open(im_path)
        seg = Image.open(seg_path)

        if self.pil_transforms is not None:
            im, seg = self.pil_transforms(image, seg)
        if self.tensor_transforms is not None:
            im = self.tensor_transforms(im)

        index = torch.from_numpy(np.array(id)).long()
        image = torch.from_numpy(np.array(image)).long()
        seg = torch.from_numpy(np.array(seg)).long()
        if self.crf:
            crf_input = im.clone()
            return im, seg, image, index, crf_input

        return im, seg, image, index
