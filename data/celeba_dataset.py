import os
import collections
import torch
import numpy as np

from PIL import Image
from torch.utils import data
from face_tran.utils import utils


class Celeba(data.Dataset):
    """Data loader for the celebA-HQ attributes dataset.

        :param root (str): path to dataset folder
        :param split (int): 0 is train, 1 is val
        :param attributes(int): how many attributes will be used
        :param is_transform (bool): if we want to proceed any transforms
        :param pil_transforms (obj): PIL transformations with both input and mask
        :param tensor_transforms (obj): Tensor transformations with input
    """

    def __init__(
                self,
                root,
                split="train",
                pil_transforms=None,
                tensor_transforms=None,
                attributes=40,
                weights=False,
    ):
        self.root = os.path.expanduser(root)

        if split == 0:
            self.split = "train"
        elif split == 1:
            self.split = "val"
        if weights:
            path = os.path.join(self.root, 'Anno')
            self.weights = torch.DoubleTensor(utils.weights_to_sample(path))
        self.classes = attributes
        path2class_dic = os.path.join(self.root, 'Anno', "new_attr_dic.txt")
        self.class_dic = utils.load_dict_from_file(path2class_dic)
        self.files = collections.defaultdict(list)
        for split in ["train", "val"]:
            path = os.path.join(self.root, 'Anno', split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        self.pil_transforms = pil_transforms
        self.tensor_transforms = tensor_transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_ = self.files[self.split][index]
        im_name = data_.split(',')[0]
        lbl = [int(x) for x in data_.split(',')[1:]]
        im_path = os.path.join(self.root, "img_celeba", im_name)
        im = Image.open(im_path)

        if self.pil_transforms is not None:
            im = self.pil_transforms(im)
        if self.tensor_transforms is not None:
            im = self.tensor_transforms(im)
        lbl = torch.from_numpy(np.array(lbl)).float()

        return im, lbl


def main(root_path, split, pil_transforms):
    import cv2
    import torchvision.transforms as transforms
    import numpy as np
    from face_tran.utils.obj_factory import obj_factory

    pil_transforms = [obj_factory(t) for t in pil_transforms] if pil_transforms is not None else []
    pil_transforms = transforms.Compose(pil_transforms)
    dataset = Celeba(root=root_path, split=split, pil_transforms=pil_transforms)
    for img, label in dataset:
        img = np.array(img)[:, :, ::-1].copy()
        cv2.imshow('img', img)
        print('label = ' + str(label))
        cv2.waitKey(0)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('image_list_dataset')
    parser.add_argument('root_path', help='paths to dataset root directory')
    parser.add_argument('-s', '--split', default=0, type=int, help='0 is train, 1 is val')
    parser.add_argument('-pt', '--pil_transforms', default=None, type=str, nargs='+', help='PIL transforms')
    args = parser.parse_args()

    main(args.root_path, args.split, args.pil_transforms)
