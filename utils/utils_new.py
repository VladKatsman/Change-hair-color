import cv2 as cv
import numpy as np
import random
import torch
import os
import face_tran.utils.utils as tran_utils
import face_alignment
import gender_guesser.detector as gdetect
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from face_tran.utils.obj_factory import obj_factory
from face_alignment.api import FaceAlignment


def alpha_blend(input_image, segmentation_mask, alpha=0.5, weights=1.0):
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


class RunningScore(object):
    """Used to compute our main metrics
        - overall accuracy
        - mean accuracy
        - mean IU
        - fwavacc

        It's better to be used at validation time only,
        because code is not optimized for Tensors and uses numpy
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


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


def crop_img_with_padding(img, bbox, seg=None):
    left = -bbox[0] if bbox[0] < 0 else 0
    top = -bbox[1] if bbox[1] < 0 else 0
    right = bbox[0] + bbox[2] - img.shape[1] if (bbox[0] + bbox[2] - img.shape[1]) > 0 else 0
    bottom = bbox[1] + bbox[3] - img.shape[0] if (bbox[1] + bbox[3] - img.shape[0]) > 0 else 0
    if any((left, top, right, bottom)):
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_REPLICATE)
        if seg is not None:
            seg = cv.copyMakeBorder(seg, top, bottom, left, right, cv.BORDER_CONSTANT)
        bbox[0] += left
        bbox[1] += top
    changes = [left, top, right, bottom]

    return img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], \
           seg[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]], changes


def scale_bbox(bbox, scale=2., square=True):
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


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

    left = left if left > 0 else 0
    top = top if top > 0 else 0
    right = right - changes[0] - changes[2] + left
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


def writer(root, lst, name="list"):
    """ Write list to .txt file

    :param root: Path to dir where we want to save file
    :param name: name of the txt file
    :param lst: List to write
    """
    path = os.path.join(root, '{}.txt'.format(name))
    with open(path, 'w') as t:
        for item in lst:
            t.write("%s\n" % item)
    print("List was successfully saved at {} file".format(path))
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
                                    #####################################
                                    ###        FACE OPERATIONS        ###
                                    #####################################


class FaceEngine(object):
    def __init__(self, det_model_path=None, lms_model_path=None,  seg_model_path=None,
                 gpus=None, cpu_only=None, max_size=640, batch_size=8,
                 conf_threshold=0.5, nms_threshold=0.4, verbose=0):

        self.max_size = max_size
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.verbose = verbose
        self.device = tran_utils.set_device(gpus, cpu_only)

        # Load face detection model
        if det_model_path is not None:
            print('Loading face detection model: "' + os.path.basename(det_model_path) + '"...')
            self.detection_net = torch.jit.load(det_model_path, map_location=self.device)
            if self.detection_net is None:
                raise RuntimeError('Failed to load face detection model!')

        # Load face landmarks model
        if lms_model_path is not None:
            print('Loading face landmarks model: "' + os.path.basename(lms_model_path) + '"...')
            self.landmarks_net = torch.jit.load(lms_model_path, map_location=self.device)
            if self.landmarks_net is None:
                raise RuntimeError('Failed to load face landmarks model!')

        # Load face segmentation model
        if seg_model_path is not None:
            print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
            if seg_model_path.endswith('.pth'):
                checkpoint = torch.load(seg_model_path)
                self.segmentation_net = obj_factory(checkpoint['arch']).to(self.device)
                self.segmentation_net.load_state_dict(checkpoint['state_dict'])
            else:
                self.segmentation_net = torch.jit.load(seg_model_path, map_location=self.device)
            if self.segmentation_net is None:
                raise RuntimeError('Failed to load face segmentation model!')
            self.segmentation_net.eval()

    def remove_neck_landmarks(self, images_path, segs_path, root):
        """ Removes neck segmentation according to face landmarks model of the FaceEngine module

        Args:
            images_path: list of pathes to images
            segs_path: list of pathes to segmentation masks

        Returns:
            Segmentation masks without neck part and saves them as root/label/name.png
        """
        # I. Set 2D landmarks detector and gender detector for lfw dataset
        lm_detector = FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        d = gdetect.Detector()

        for i in tqdm(range(len(images_path))):
            name = segs_path[i].split('/')[-1]

            # II. Open IMAGE and segmentation MASK
            source_img = cv.imread(images_path[i])
            source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
            source_seg = Image.open(segs_path[i])

            # III. Find LANDMARKS
            landmarks = lm_detector.get_landmarks_from_image(source_img)[0]
            # in order to remove ears as well
            left_b = [landmarks[0][0], landmarks[0][1] - (landmarks[2][1] - landmarks[0][1])]
            right_b = [landmarks[16][0], landmarks[16][1]-(landmarks[14][1] - landmarks[16][1])]
            lm = landmarks[:17]

            # IV. Get coordinates of points to create separation line (border)
            h, w = source_seg.size
            left_edge = [0, left_b[1]]
            right_edge = [w-1, right_b[1]]
            points_neck = left_edge + left_b + list(lm.ravel()) + right_b + right_edge
            points_beard = left_edge + left_b + right_b + right_edge

            # V. Get starting points for filler algorithm
            left = [left_edge[0], left_edge[1]+1]
            right = [right_edge[0], right_edge[1]+1]

            # VI. Draw border line
            source_seg = np.array(source_seg).astype(np.uint8)
            if len(np.unique(source_seg)) != 3:
                source_seg[source_seg==1] = 0
                source_seg[source_seg==2] = 1
            mask_neck = source_seg.copy()
            # mask_neck[mask_neck == 2] = 0
            mask_neck = Image.fromarray(mask_neck)
            draw = ImageDraw.ImageDraw(mask_neck)
            draw.line(points_neck, 255, 0)

            # VII. Create a mask without neck
            mask_neck = np.array(mask_neck)
            mask_no_neck = fill_underline(mask_neck, h, w, left, right)
            face_mask = source_seg.copy()

            # VIII. Remove redundant features
            first_name = name.split('_')[0]
            sex = d.get_gender(first_name)
            if len(np.unique(source_seg)) == 3:
                if sex == 'male':
                    # Cut beards and mustache
                    mask_beard = source_seg.copy()
                    mask_beard = Image.fromarray(mask_beard)

                    # draw separation line
                    draw = ImageDraw.ImageDraw(mask_beard)
                    draw.line(points_beard, 255, 0)

                    # fill all the area with 1 label
                    mask_beard = np.array(mask_beard)
                    mask_no_beard = fill_underline(mask_beard, h, w, left, right, fill=1)

                    # remove mustache
                    face_mask[mask_no_beard == 1] = 1

                    # remove beard and neck
                    face_mask[mask_no_neck == 0] = 0

                else:
                    # remove neck only for women
                    face_mask[mask_no_neck == 0] = 0
                    face_mask[source_seg == 2] = 2

            else:
                face_mask[mask_no_neck == 0] = 0

            # IX. Add palette
            mask_f = Image.fromarray(face_mask)
            if len(np.unique(face_mask)) == 3:
                mask_f = mask_f.convert('P', palette=Image.ADAPTIVE, colors=3)
                reverse_colors = np.array(mask_f)
                reverse_colors[reverse_colors == 0] = 3
                reverse_colors[reverse_colors == 2] = 0
                reverse_colors[reverse_colors == 3] = 2
            else:
                mask_f = mask_f.convert('P', palette=Image.ADAPTIVE, colors=2)
                reverse_colors = np.array(mask_f)
                reverse_colors[reverse_colors == 1] = 4
                reverse_colors[reverse_colors == 0] = 1
                reverse_colors[reverse_colors == 4] = 0
            mask_f = Image.fromarray(reverse_colors, mode='P')
            mask_f.putpalette([
                0, 0, 0,  # index 0 is black (background)
                0, 255, 0,  # index 1 is green (face)
                255, 0, 0,  # index 2 is red (hair)
            ])

            # X. Save results
            path = "{}/Masks/{}".format(root, name)
            mask_f.save(path)
