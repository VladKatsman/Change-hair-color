import math
import numbers
import numpy as np
import random
import cv2
import imgaug as iaa

from torch.nn import functional as tf
from torchvision.transforms import functional
from PIL import Image
from imgaug import augmenters as iaa


# Joint transformations are adopted from
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/augmentations/augmentations.py


class RandomResizedCropPair(object):
    """Crop the given PIL Image and It's segmentation mask at the same random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant',
                 scale=(0.5, 2.0), ratio=(1.0, 1.0), interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = functional.pad(img, self.padding, self.fill, self.padding_mode)
            lbl = functional.pad(lbl, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = functional.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            lbl= functional.pad(lbl, (self.size[1] - lbl.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = functional.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            lbl = functional.pad(lbl, (0, self.size[0] - lbl.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        lbl = functional.resized_crop(lbl, i, j, h, w, self.size, self.interpolation)

        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ComposePair(object):
    """ Compose joint transformations for segmentation problems

        For input pair (img, lbl) where lbl is Segmentation Mask of img
    """
    def __init__(self, augmentations):

        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        if type(self.augmentations) != list:
            img, mask = self.augmentations(img, mask)
        else:
            for a in self.augmentations:
                img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8)

        return img, mask


class ResizePair(object):
    """ In order to use Resize together with other pair methods"""
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        img = functional.resize(img, self.size, self.interpolation)
        mask = functional.resize(mask, self.size, Image.NEAREST)

        return img, mask


class RandomHorizontallyFlipPair(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask

class RandomHorizontallyFlipTrio(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, im):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
                im.transpose(Image.FLIP_LEFT_RIGHT)
            )
        return img, mask, im

class RotatePair(object):
    def __init__(self, p=0.5, angle=(-10, 10)):
        self.p = p
        self.angle = angle

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = random.randint(self.angle[0], self.angle[1])

            return (
                img.rotate(angle, Image.BICUBIC, False),
                mask.rotate(angle, Image.NEAREST, False),
            )
        return img, mask


def crop_img(img,  bbox, lbl=None, rescaled_size=False):
    """ Crops image, according to bboxes.

    Args:
        img: PIL image, image to crop
        bbox: list of ints, coordinates of boundary boxes
        lbl: optionally, PIL Image, crop ground truth as well
        rescaled_size: optionally, if you want output to be of the same size

    Returns:

    """
    img = np.array(img).astype(np.uint8)

    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0

    # in order to build tensor, we want to make sure, we have the same size amongst batch
    if rescaled_size:
        diff_vert = rescaled_size[0] - bbox[3] if (rescaled_size[0] - bbox[3]) > 0 else 0
        diff_hor = rescaled_size[1] - bbox[2] if (rescaled_size[1] - bbox[2]) > 0 else 0
        bbox[2] += diff_hor
        bbox[3] += diff_vert
        right += diff_hor
        bottom += diff_vert
    if any((left, top, right, bottom)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)   # <-- cv2.BORDER_REPLICATE
    if lbl is not None:
        lbl = np.array(lbl).astype(np.uint8)
        if any((left, top, right, bottom)):
            lbl = cv2.copyMakeBorder(lbl, top, bottom, left, right, cv2.BORDER_CONSTANT)
            bbox[0] += left
            bbox[1] += top

        return (img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]],
                lbl[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
    else:
        bbox[0] += left
        bbox[1] += top

    return img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]


def decrop_mask(original_mask, cropped_mask, bbox, changes):
    """ Removes all changes being made by crop_img and scale_bbox functions

        In order to add face class to original Segmentation mask
        we need to retrieve original fragment being cut from mtcnn bboxes
        with a predicted face label upon it.

    :param img: img which changes have to be removed
    :param changes: padding to be removed, output of the "crop_img" function
    :param bbox: bboxes being used to crop an image (output of "scale_bbox" function)
    :return: original Segmentation Mask with face class
    """
    # locating cropped_region on the original_mask
    left, top, right, bot = bbox
    
    x_max, y_max = original_mask.shape
    left = left if left > 0 else 0
    top = top if top > 0 else 0
    right = right - changes[0] - changes[2]  + left
    bot = bot - changes[1] - changes[3] + top
    
    # remove padding of cropped region, if bboxes region was relatively big after 2x scaling
    l = changes[0]
    t = changes[1]
    r = bbox[2] - changes[2]
    b = bbox[3] - changes[3]

    mask_no_pad = cropped_mask[t:b, l:r]

    # insert cropped mask with face label on the same location of the original mask, where crop being cut from
    original_mask[top:bot, left:right] = mask_no_pad

    return original_mask


def decrop_image(original_img, cropped_img, bbox, changes):
    """ Removes all changes being made by crop_img and scale_bbox functions

        In order to add face class to original Segmentation mask
        we need to retrieve original fragment being cut from mtcnn bboxes
        with a predicted face label upon it.

    :param img: img which changes have to be removed
    :param changes: padding to be removed, output of the "crop_img" function
    :param bbox: bboxes being used to crop an image (output of "scale_bbox" function)
    :return: original Segmentation Mask with face class
    """
    # locating cropped_region on the original_mask
    left, top, right, bot = bbox

    left = left if left > 0 else 0
    top = top if top > 0 else 0
    right = right - changes[0] - changes[2] + left
    bot = bot - changes[1] - changes[3] + top

    # remove padding of cropped region, if bboxes region was relatively big after 2x scaling
    l = changes[0]
    t = changes[1]
    r = bbox[2] - changes[2]
    b = bbox[3] - changes[3]

    crop_no_pad = cropped_img[t:b, l:r, :]

    # insert cropped mask with face label on the same location of the original mask, where crop being cut from
    original_img[top:bot, left:right, :] = crop_no_pad

    return original_img


def preprocess_bbox(bbox):
    """ bboxes should be in the next format:
        [min_x, min_y, width, height]
    :param bbox: mtcnn bbox
    :return: bbox in format we use further
    """
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    return np.array([min_x, min_y, width, height])


def scale_bbox(bbox, scale=2., square=True):


    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def align_crop(img, landmarks, bbox, scale=2.0, square=True):
    # Rotate image for horizontal eyes
    if landmarks.shape[0] != 5:
        landmarks = np.asarray([landmarks.reshape(2,5)[:,x] for x in range(5)], int)
    right_eye_center = landmarks[0]
    left_eye_center = landmarks[1]
    eye_center = np.round(np.mean(landmarks[:2], axis=0)).astype(int)
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.)
    output = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Adjust landmarks
    new_landmarks = np.concatenate((landmarks, np.ones((66, 1))), axis=1)
    new_landmarks = new_landmarks.dot(M.transpose())

    # Scale bounding box
    bbox_scaled = scale_bbox(bbox, scale, square)

    # Crop image
    output, new_landmarks = crop_img(output, new_landmarks, bbox_scaled)

    return output, new_landmarks, angle


class LandmarksTransform(object):
    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to transform.
            landmarks (numpy.ndarray): Array of face landmarks (68 X 2)
            bbox (numpy.ndarray): Face bounding box (4,)

        Returns:
            Tensor: Converted image, landmarks, and bounding box.
        """
        return img, landmarks, bbox


class FaceAlignCrop(LandmarksTransform):
    """Aligns and crops pil face images.

    Args:
        bbox_scale (float): Multiplier factor to scale tight bounding box
        bbox_square (bool): Force crop to be square.
        align (bool): Toggle face alignment using landmarks.
    """

    def __init__(self, bbox_scale=2.0, bbox_square=True, align=False):
        self.bbox_scale = bbox_scale
        self.bbox_square = bbox_square
        self.align = align

    def __call__(self, img, landmarks, bbox):
        """
        Args:
            img (PIL Image): Face image to align and crop.
            landmarks (numpy array): Face landmarks
            bbox (numpy array): Face tight bounding box

        Returns:
            PIL Image: Rescaled image.
        """
        img = np.array(img).copy()
        if self.align:
            img, landmarks = align_crop(img, landmarks, bbox, self.bbox_scale, self.bbox_square)
        else:
            bbox_scaled = scale_bbox(bbox, self.bbox_scale, self.bbox_square)
            img, landmarks = crop_img(img, landmarks, bbox_scaled)

        img = Image.fromarray(img)

        return img, landmarks, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_scale={0}, bbox_square={1}, align={2})'.format(
            self.bbox_scale, self.bbox_square, self.align)


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation)),
            mask,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf)), mask


class ImgTransform:
    """ Combination of augmentors from https://github.com/aleju/imgaug/tree/master/imgaug/augmenters """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GammaContrast((0.75, 1.25))),
            iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True),
            iaa.Affine(rotate=(-35, 35))
        ])

    def __call__(self, img):
        img = np.array(img)

        return self.aug.augment_image(img)

class GeoTransform:
    """ Combination of augmentors from https://github.com/aleju/imgaug/tree/master/imgaug/augmenters """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Affine(rotate=(-35, 35))
        ])

    def __call__(self, img, lbl):
        img = np.array(img)
        lbl = np.array(lbl)
        det = self.aug.to_deterministic()
        return det.augment_image(img), det.augment_image(lbl)