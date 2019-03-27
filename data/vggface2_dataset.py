import os
import collections
import torch
import numpy as np

from PIL import Image
from torch.utils import data
from face_tran.utils import seg_transforms


class VGGFace2(data.Dataset):
    """Data loader for the faces/hair semantic segmentation dataset.

        Meanwhile, consists of Figaro

        :param root (str): path to dataset folder
        :param split (int): 0 is train, 1 is val
        :param bboxes(array): name of the file with bboxes in the root dir
        :param augmentations (obj): augmentations we want to use
        :param pil_transforms (obj): PIL transformations with both input and mask
        :param tensor_transforms (obj): Tensor transformations with input

    """

    def __init__(
                self,
                root,
                split="train",
                pil_transforms=None,
                tensor_transforms=None,
                visual_augmentations=None,
                geo_augmentations=None,
                test=False,
                bboxes=False,
                mask=False,
    ):
        self.root = os.path.expanduser(root)

        if split == 0:
            self.split = "train"
        elif split == 1:
            self.split = "val"
        self.test = test
        self.mask = mask
        self.n_classes = 3
        self.files = collections.defaultdict(list)
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
        self.visual_augmentations = visual_augmentations
        self.geo_augmentations = geo_augmentations
        if bboxes:
            path_bbox = os.path.join(self.root, "bboxes_{}.npy".format(self.split))
            bboxes = np.load(path_bbox)
            self.bboxes = bboxes
        else:
            self.bboxes = None
        self.pil_transforms = pil_transforms
        self.tensor_transforms = tensor_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_name = self.images[index]
        seg_name = self.segs[index]
        im_path = os.path.join(self.root, im_name)
        lbl_path = os.path.join(self.root, seg_name)
        im = Image.open(im_path)

        if self.test:
            lbl = Image.open(im_path)  # in inference mode we dont need mask, we need an original image
            h, w = im.size
            for t in self.pil_transforms:
                im = t(im)
                lbl = t(lbl)
            im = self.tensor_transforms(im)
            lbl = torch.from_numpy(np.array(lbl)).long()

            if not self.mask:
                return im, lbl
            else:
                return im, im_name, h, w

        lbl = Image.open(lbl_path)
        if self.visual_augmentations is not None:
            im = self.visual_augmentations(im)
            im = Image.fromarray(im)

        if self.geo_augmentations is not None:
            aug = self.geo_augmentations.transforms[0]
            im, lbl = aug(im, lbl)
            im = Image.fromarray(im)
            lbl = Image.fromarray(lbl)

        if self.bboxes is not None:
            if len(self.bboxes[index]) != 0:
                bbox = self.bboxes[index]
                # bbox = seg_transforms.preprocess_bbox(bbox) <- for bboxes in format x0,y0,x1,y1
                bbox = seg_transforms.scale_bbox(bbox, scale=2., square=True)
                im, lbl = seg_transforms.crop_img(im, bbox, np.array(lbl))
                im = Image.fromarray(im.astype(np.uint8))
                lbl = Image.fromarray(lbl.astype(np.uint8))
        if self.pil_transforms is not None:
            im, lbl = self.pil_transforms(im, lbl)
        if self.tensor_transforms is not None:
            im = self.tensor_transforms(im)
            image = im.clone()  # for validation purposes

        lbl = torch.from_numpy(np.array(lbl)).long()
        if self.split == 'val':
            return im, lbl, image

        return im, lbl
