import cv2 as cv
import numpy as np
import random
import torch
import os
import face_alignment

from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from hair_color_change.utils.obj_factory import obj_factory
from hair_color_change.utils.detection_utils import detect
from face_alignment.api import FaceAlignment
from face_tran.utils.my_utils import writer


def set_device(gpus=None, cpu_only=None):
    use_cuda = torch.cuda.is_available() if cpu_only is None else not cpu_only
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


def process_graph_cut(prediction_mask, image, bbox, sizes=None):
    """ To increase robust area of segmentation of the hair label from output of UNet model

    Args:
        prediction_mask(array): hard mask of output of UNet model
        image (array): input Image
        bbox (list of int): list of bboxes with detected face
        sizes (tuple): original image size [H, W]

    Returns: binary array
    """
    # prepare input for cv.grabCut function
    label = prediction_mask.copy()
    label[label == 1] = 4  # face label have to become true background (0)
    label[label == 2] = 1  # hair label have to become true foreground (1)
    label[label == 0] = 2  # background have to become probably background (2)
    label[label == 4] = 0
    label[label == 3] = 2

    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    # in order to use cv.grabCut function go to numpy
    img = image.copy()
    if bbox is not None:
        bbox = tuple(bbox)
    try:
        cv.grabCut(img, label, bbox, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
    except:
        print("grabCut failed")
    mask = np.where((label == 2) | (label == 0), 0, 1).astype('uint8')

    #tmp
    # trying to apply fixed background
    #ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # mask2[mask2==1] = 200
    # cs2 = Image.fromarray(mask2)

    # # WaterShed
    # # basic step: binary mask
    # val, label = predictions.max(1)
    # val, label = val.squeeze(dim=0), label.squeeze(dim=0)
    # zero = torch.zeros_like(val)
    # binary_mask = torch.where(label == 2, val, zero).squeeze(dim=0)
    # binary_mask = torch.gt(binary_mask, zero)
    #
    # # vanilla
    # image2 = image.copy()
    # #tmp
    # color = 0
    # new, contours, hierarchy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # mask2[mask2==0]=125
    # mask2[mask2==1]=255
    # cv.drawContours(mask2, contours, -1, color, 1)
    #
    # # continue
    # fg = np.zeros_like(mask2)
    # fg[mask2 == 255] = 1
    # bg = np.zeros_like(mask2)
    # bg[mask2 == 125] = 1
    # unknown = cv.subtract(bg, fg)

    # TESTS
    # adv step: soft mask
    # hair = predictions.max(0)[0][2]
    # hair_mask = torch.where(label == 2, val, hair/val)
    #
    # # norm mask TEST
    # zero = torch.zeros_like(hair_mask)
    # hair_mask = torch.where(hair_mask > 0, hair_mask, zero)
    # hair_mask = hair_mask.round()
    # mean_pix = hair_mask.sum()/torch.gt(hair_mask, zero).sum()
    #
    # # threshold = max_pix//2
    # threshold = mean_pix// 2
    # hair_mask1 = torch.where((hair_mask > threshold), hair_mask, zero)
    # hair_mask1 = torch.where((hair_mask1 < mean_pix) & (hair_mask1 > threshold), mean_pix, hair_mask1)
    #
    # # threshold = max_pix//2.5
    # threshold = mean_pix // 2.5
    # hair_mask2 = torch.where((hair_mask > threshold), hair_mask, zero)
    # hair_mask2 = torch.where((hair_mask2 < mean_pix) & (hair_mask2 > threshold), mean_pix, hair_mask2)
    #
    # # TEST
    # original = binary_mask.cpu().numpy().astype(np.uint8)
    # original[original == 1] = 200
    # half = hair_mask1.cpu().numpy().astype(np.uint8)
    # third = hair_mask2.cpu().numpy().astype(np.uint8)
    #
    # imgs = [Image.fromarray(x) for x in [original, half, third]]
    # visualize = pil_grid(imgs)

    # apply color
    # max_val = hair_mask.max()
    # weights_mask = hair_mask / max_val
    # weights_mask = weights_mask.repeat(1, predictions.shape[1], 1, 1)

    return mask


def apply_color_mask(mask, color):
    """ Blends image with soft colour mask

    Args:
        colors (list): XYZ coded color scheme (RGB, BGR, HSV..)
        mask (array): postprocessed mask of values from 0 to 1
        smooth_mask (array): adds some noise to the image

    Returns:
        Mask of specific color
    """
    # mask = mask/255

    X = mask * color[0]
    Y = mask * color[1]
    Z = mask * color[2]

    hsv_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    hsv_mask[:, :, 0] = X  # / 255.0
    hsv_mask[:, :, 1] = Y  # / 255.0
    hsv_mask[:, :, 2] = Z  # / 255.0
    hsv_mask = hsv_mask.astype(np.uint8)

    return hsv_mask


def hsv_blend(source_hsv, desaturated_color_hsv, soft_mask):
    """Alpha Blending utility to overlay RGB masks on RBG images

    Args:
        source_hsv: array, source image in hsv format
        desaturated_color_hsv: array, desaturated source image with specified color in hsv format
        soft_mask: soft hair mask

    Returns:
        blended image, PIL.Image in rgb format
    """
    mask = soft_mask.copy()
    # test
    mask = mask[:, :, np.newaxis]
    hsv = source_hsv.copy()
    # test
    hsv[:, :, :1] = hsv[:, :, :1] * (1-mask) + desaturated_color_hsv[:, :, :1] * mask

    blend = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    blend = Image.fromarray(blend)

    return blend


def alpha_blend(input_image, colored_mask, segmentation_mask):
    """Alpha Blending utility to overlay RGB masks on RBG images

    Args:
        input_image: array
        color_mask: array segmentation mask of some color
        segmentation_mask: array
        opening_mask: array
        alpha (float): more val -> more weight to input image,
                       less val -> more weight to segmentation mask color
        weights: PIL.Image or np.ndarray, add smoothness to mask colors

    Returns:

    """
    image = input_image.copy()
    mask = segmentation_mask.copy()[:, :, np.newaxis]
    blend = image * (1-mask) + colored_mask * mask
    blend = Image.fromarray(blend.astype(np.uint8))

    return blend


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


def decode_segmap(label_mask, n_classes=3, label_colours=get_labels()):
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


def alpha_blend_rgb(input_image, segmentation_mask, alpha=0.5, weights=1.0):
    """Alpha Blending utility to overlay RGB masks on RBG images

    :param input_image:
    :param segmentation_mask:
    :param alpha: is a  value
    :param

    Args:
        input_image: PIL.Image or np.ndarray
        segmentation_mask: PIL.Image or np.ndarray
        alpha (float): more val -> more weight to input image,
                       less val -> more weight to segmentation mask color
        weights: PIL.Image or np.ndarray, add smoothness to mask colors

    Returns:

    """
    if type(input_image) != np.ndarray:
        input_image = np.array(input_image).astype(np.uint8)
    if type(segmentation_mask) != np.ndarray:
        segmentation_mask = np.array(segmentation_mask).astype(np.uint8)
    blended = input_image * alpha + segmentation_mask * (1-alpha) * weights

    return blended


def fill_underline(mask, h, w, left, right, border=255, fill=0):
    """ Fills an area of segmentation mask below border line with zeros

    Args:
        mask: segmentation mask with border line to fill
        h: height of mask
        w: width of mask
        left: left side start point
        right: right side start point
        fill: value to fill

    Returns:
        Segmentation mask where all values below border line are equal to zero
    """
    # start from the left
    y, x = int(left[0]), int(left[1])
    stop = h
    while x < stop:
        while y < w and mask[x, y] != 255:
            mask[x, y] = fill
            y += 1
        # if we find line without border at all, continue to second part
        if y == w:
            stop = x
        else:
            y = 0
            x += 1
    # start from the right
    y, x = int(right[0]), int(right[1])
    while x < h:
        while y >= 0 and mask[x, y] != 255:
            mask[x, y] = fill
            y -= 1
        y = w-1
        x += 1

    return mask
                                ###########################
                                ###      INFERENCE      ###
                                ###########################


def pil_grid(images, max_horiz=np.iinfo(int).max):
    """ Concatenates images horizontally

    :param images: list of images to concatenate
    :param max_horiz: default is horizontal, 1 for vertical
    """
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid
                                    ##########################
                                    ###    CROPS RUTINE    ###
                                    ##########################


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].sub_(1).fmod_(float(hm.size(3))).add_(1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    # Finetune
    hm_padded = torch.nn.functional.pad(hm, (1, 1, 1, 1), 'replicate')
    ind0 = torch.arange(hm.shape[0], device=hm.device).repeat(hm.shape[1], 1).permute(1, 0).contiguous().view(-1)
    ind1 = torch.arange(hm.shape[1], device=hm.device).repeat(hm.shape[0])
    ind2 = preds[:, :, 1].long().view(-1) + 1
    ind3 = preds[:, :, 0].long().view(-1) + 1
    diffx = hm_padded[(ind0, ind1, ind2, ind3 + 1)] - hm_padded[(ind0, ind1, ind2, ind3 - 1)]
    diffy = hm_padded[(ind0, ind1, ind2 + 1, ind3)] - hm_padded[(ind0, ind1, ind2 - 1, ind3)]
    preds[:, :, 0].view(-1).add_(diffx.sign_().mul_(.25))
    preds[:, :, 1].view(-1).add_(diffy.sign_().mul_(.25))

    preds.add_(-.5).mul_(4.)

    return preds


def get_main_bbox(bboxes, img_size):
    if len(bboxes) == 0:
        return None

    # Calculate frame max distance and size
    img_center = np.array([img_size[1], img_size[0]]) * 0.5
    max_dist = 0.25 * np.linalg.norm(img_size)
    max_size = 0.25 * (img_size[0] + img_size[1])

    # For each bounding box
    scores = []
    for bbox in bboxes:
        # Calculate center distance
        bbox_center = bbox[:2] + bbox[2:] * 0.5
        bbox_dist = np.linalg.norm(bbox_center - img_center)

        # Calculate bbox size
        bbox_size = bbox[2:].mean()

        # Calculate central ratio
        central_ratio = 1.0 if max_size < 1e-6 else (1.0 - bbox_dist / max_dist)
        central_ratio = np.clip(central_ratio, 0.0, 1.0)

        # Calculate size ratio
        size_ratio = 1.0 if max_size < 1e-6 else (bbox_size / max_size)
        size_ratio = np.clip(size_ratio, 0.0, 1.0)

        # Add score
        score = (central_ratio + size_ratio) * 0.5
        scores.append(score)

    return bboxes[np.argmax(scores)]


def crop_img_with_padding(img, bbox):
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    changes = [left, top, right, bottom]

    if any((left, top, right, bottom)):
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT)  # cv.BORDER_REPLICATE
        bbox[0] += left
        bbox[1] += top

    return img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], changes


def scale_bbox(bbox, scale=2., square=True):
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def scale_bbox_square(bbox, source_size, scale=2.):
    h, w = source_size
    bbox_center = bbox[:2] + bbox[2:] / 2
    side_size = np.min((h - bbox_center[0] * 0.8, bbox[2] * scale))
    top, bot = int(bbox_center[1] - side_size), int(side_size * 2)
    left, right = int(bbox_center[0] - side_size), int(side_size * 2)

    return np.array([left, top, right, bot])


# def insert_align(sizes, mask, center):


def decrop_mask(sizes, mask, bbox, changes):
    """ Makes predicted mask to be consistent with original image

    Args:
        sizes (array): shape of original image shape
        mask (array): postprocessed mask to be rescaled
        changes (list of ints): padding to be removed, output of the "crop_img" function
        bbox (list of ints): bboxes being used to crop an image (output of "scale_bbox" function)

    Return:
        Segmentation Mask which consistent (size-wise) with original image
    """
    # locating cropped_region on the original_mask
    left, top, right, bot = bbox

    right = right - changes[0] - changes[2] + left
    bot = bot - changes[1] - changes[3] + top

    # remove padding of cropped region, if bboxes region was relatively big after 2x scaling
    l = changes[0]
    t = changes[1]
    r = bbox[2] - changes[2]
    b = bbox[3] - changes[3]

    mask_no_pad = mask[t:b, l:r]

    # insert cropped mask with face label on the same location of the original mask, where crop being cut from
    mask = np.zeros(sizes)
    mask[top:bot, left:right] = mask_no_pad

    return mask.astype(np.uint8)


def align_remove(aligned_img, restore_info):

    # get variables
    angle, eye_center = restore_info

    # rotate mask to original angle, so minus angle
    M = cv.getRotationMatrix2D(tuple(eye_center), -angle, 1.)
    if len(aligned_img.shape) == 2:
        flag = cv.INTER_NEAREST
    else:
        flag = cv.INTER_CUBIC
    mask_of_original_scale = cv.warpAffine(aligned_img, M, (aligned_img.shape[1], aligned_img.shape[0]), flags=flag)

    return mask_of_original_scale


def align(img, landmarks):
    # Rotate image for horizontal eyes
    right_eye_center = np.round(np.mean((
        landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]), axis=0)).astype(int)
    left_eye_center = np.round(np.mean((
        landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]), axis=0)).astype(int)
    eye_center = np.round(np.mean((right_eye_center, left_eye_center), axis=0)).astype(int)
    dy = right_eye_center[0] - left_eye_center[0]
    dx = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dx, dy))  # - 180  <-- do not need

    M = cv.getRotationMatrix2D(tuple(eye_center), angle, 1.)
    aligned_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC)

    # Keep variables to restore original mask coordiantes
    restore_info = [angle, eye_center]

    return aligned_img, restore_info

                                        ##########################
                                        ###        MISC        ###
                                        ##########################


def txt2list(path):
    """ Reads .txt file and saves it as list"""
    l = []
    with open(path) as a:
        for line in a:
            l.append(line.rstrip())
    return l

                                    #####################################
                                    ###       TENSOR OPERATIONS       ###
                                    #####################################


def rgb2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def bgr2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(o, normalize) for o in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2rgb(img_tensor):
    output_img = unnormalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def tensor2bgr(img_tensor):
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]

    return output_img
