import os
import numpy as np
import cv2 as cv
import torch
from glob import glob
from tqdm import tqdm
from hair_color_change.utils import utils, detection_utils
from PIL import Image


class FaceEngine(object):
    def __init__(self, det_model_path=None, lms_model_path=None, reenactment_model_path=None, seg_model_path=None,
                 gpus=None, cpu_only=None, max_size=640, conf_threshold=0.5, nms_threshold=0.4, verbose=0, mode='hair'):

        self.max_size = max_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.verbose = verbose
        self.device, self.gpus = utils.set_device(gpus, cpu_only)
        if mode == "hair":  # 2 means hair, 1 means face, 0 means background
            self.mode = 2

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
            # self.landmarks_net.eval()

        # Load face reenactment model
        if reenactment_model_path is not None:
            print('Loading face reenactment model: "' + os.path.basename(reenactment_model_path) + '"...')
            self.reenactment_net = torch.jit.load(reenactment_model_path, map_location=self.device)
            if self.reenactment_net is None:
                raise RuntimeError('Failed to load face reenactment model!')

        # Load face segmentation model
        if seg_model_path is not None:
            print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
            if seg_model_path.endswith('.pth'):
                checkpoint = torch.load(seg_model_path)
                self.segmentation_net = utils.obj_factory(checkpoint['arch']).to(self.device)
                self.segmentation_net.load_state_dict(checkpoint['state_dict'])
            else:
                self.segmentation_net = torch.jit.load(seg_model_path, map_location=self.device)
            if self.segmentation_net is None:
                raise RuntimeError('Failed to load face segmentation model!')
            self.segmentation_net.eval()

    def change_hair_color(self, source_path, output_path, size=256, color=None):
        """ Change color of hairs
        Args:
            source_path: path to the image to change color
            output_path: path to dir to save result

        Returns:
            Image with hairs of different color from the original one
        """
        torch.set_grad_enabled(False)

        # 1. Open IMAGE and get original name (suffix)
        source_img = cv.imread(source_path)
        source_size = tuple([int(x) for x in source_img.shape[:2]])
        source_size = source_size[::-1]
        source_bgr = cv.addWeighted( source_img, 1, source_img, 0, 1)
        source_rgb = source_bgr[:, :, ::-1]
        name = os.path.split(source_path)[-1]

        # 2. Check whether IMAGE is grayscale or not
        if len(source_bgr.shape) != 3:
            source_bgr = cv.cvtColor(source_bgr, cv.COLOR_GRAY2BGR)

        if source_bgr is None:
            print('Failed to load source image: ' + source_path)
            return 3

        # 3. Pass through box detector
        source_bbox = detection_utils.detect(source_bgr, self.detection_net, self.device, self.max_size)
        restore_info = 0
        if source_bbox is not None:

            # choose main bbox
            if len(source_bbox.shape) > 1:
                source_bbox = utils.get_main_bbox(source_bbox, (source_size[1], source_size[0]))

            # scale bbox and crop face with an increased crop size (=2.)
            source_bbox_scaled = utils.scale_bbox(source_bbox, scale=2.5).astype(int)
            source_bgr_processed, changes = utils.crop_img_with_padding(source_bgr, source_bbox_scaled)
            h, w, _ = source_bgr_processed.shape  # in order to insert that part back in the image

            # find landmarks
            source_bgr_processed = cv.resize(source_bgr_processed, (size, size), interpolation=cv.INTER_CUBIC)  # source_bgr
            source_tensor = utils.bgr2tensor(source_bgr_processed, normalize=True)
            source_tensor = source_tensor.to(self.device)
            landmarks_hmpred_tensor = self.landmarks_net(source_tensor)

            # if we found landmarks, align image
            if landmarks_hmpred_tensor is not None:  # TODO check what we return if prediction failed
                lm = utils.get_preds_fromhm(landmarks_hmpred_tensor).squeeze(0).cpu().numpy().astype(np.uint8)
                lm = lm.astype(np.uint16)
                source_bgr_processed, restore_info = utils.align(source_bgr_processed, lm)
                # h, w = source_size

        else:
            print('Failed to detect a face in the source image!')
            print('Using source image without crop')
            source_bgr_processed = source_bgr

        # 4. Preprocessing before segmentation
        source_bgr_processed = cv.resize(source_bgr_processed, (size, size), interpolation=cv.INTER_CUBIC)
        source_tensor = utils.bgr2tensor(source_bgr_processed, normalize=True)
        source_tensor = source_tensor.to(self.device)

        # 5. Run segmentation model to get hair and face segments
        segmentation_tensor = self.segmentation_net(source_tensor)
        pred = segmentation_tensor.max(1)[1].cpu().numpy().astype(np.uint8).squeeze(0)
        face_mask = pred.copy()
        face_mask[face_mask != 1] = 0

        pred[pred == 1] = 0
        pred[pred == 2] = 255

        # remove align operation if needed
        if restore_info != 0:
            pred = utils.align_remove(pred, restore_info)
            face_mask = utils.align_remove(face_mask, restore_info)

        # 6. Upscale masks to original sizes
        if source_bbox is not None:
            pred = cv.resize(pred, (h, w), interpolation=cv.INTER_CUBIC)
            face_mask = cv.resize(face_mask, (h, w), interpolation=cv.INTER_CUBIC)

            # insert predicted segment back to the same place it was cut from
            if restore_info != 0:
                pred = utils.decrop_mask((source_size[1], source_size[0]), pred, source_bbox_scaled, changes)
                face_mask = utils.decrop_mask((source_size[1], source_size[0]), face_mask, source_bbox_scaled, changes)
        else:
            pred = cv.resize(pred, (source_size[0], source_size[1]), interpolation=cv.INTER_CUBIC)
            face_mask = cv.resize(face_mask, (source_size[0], source_size[1]), interpolation=cv.INTER_CUBIC)
        # 7. Mask processing, erode and add gaussian blur for smooth shape

        # smooth masks: FACE and HAIR
        face_mask = cv.GaussianBlur(face_mask, (51, 51), 0)
        pred = cv.blur(pred, (31, 31), 0)
        pred[face_mask > 0] = 0
        mask = pred.copy()
        mask = Image.fromarray(mask)
        pred = pred / 255

        # 8. Desaturate image
        desaturated = source_rgb.copy()
        desaturated = cv.cvtColor(desaturated, cv.COLOR_RGB2GRAY)
        desaturated = cv.cvtColor(desaturated, cv.COLOR_GRAY2RGB)
        desaturated = cv.cvtColor(desaturated, cv.COLOR_RGB2HSV_FULL)

        # 9. choose HSV color, meanwhile simple colors with a step of 50 for HUE
        color1 = desaturated.copy()
        color2 = desaturated.copy()
        color3 = desaturated.copy()
        color4 = desaturated.copy()
        colors = [color1, color2, color3, color4]
        i = 1
        for color in colors:
            color[:, :, 0] += i * 50
            color[:, :, 1] = 130
            # color[:, :, 2] = np.where((color[:, :, 2] < 100), color[:, :, 2] + 100, color[:, :, 2])
            i += 1
        colors = [cv.cvtColor(x, cv.COLOR_HSV2RGB) for x in colors]

        # 10. blend color with hair segment of original image
        blend1 = utils.alpha_blend(source_rgb, colors[0], pred)
        blend2 = utils.alpha_blend(source_rgb, colors[1], pred)
        blend3 = utils.alpha_blend(source_rgb, colors[2], pred)
        blend4 = utils.alpha_blend(source_rgb, colors[3], pred)

        # 11. save results
        results = [Image.fromarray(source_rgb.astype(np.uint8)), mask, blend1, blend2, blend3, blend4]
        grid = utils.pil_grid(results)
        grid.save("{}/{}".format(output_path, name))


def main(source_path, output_path, det_model_path='models/face_detection_s3fd.pt',
         lms_model_path = 'models/2DFAN-4_hm.pt',
         seg_model_path='models/unet_face_segmentation_256.pth',
         max_size=640, gpus=None, cpu_only=None, verbose=0):

    # set engine on
    hair_engine = FaceEngine(det_model_path=det_model_path, lms_model_path=lms_model_path,
                             seg_model_path=seg_model_path, gpus=gpus, cpu_only=cpu_only, max_size=max_size)

    # hair_engine.change_hair_color(source_path, output_path)

    pathes = glob("{}/*".format(source_path))
    pathes.sort()
    for path in tqdm(pathes):
        hair_engine.change_hair_color(path, output_path)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('Swap')
    parser.add_argument('source', metavar='IMAGE',
                        help='path to source image')
    parser.add_argument('-o', '--output', default=None, metavar='PATH',
                        help='output path')
    parser.add_argument('-d', '--det_model', default='models/face_detection_s3fd.pt', metavar='PATH',
                        help='path to face detection model file')
    parser.add_argument('-l', '--lms_model', default='models/2DFAN-4_hm.pt', metavar='PATH',
                        help='path to landmarks detection model file')
    parser.add_argument('-s', '--seg_model', default='models/unet_face_segmentation_256.pth', metavar='PATH',
                        help='path to face segmentation model file')
    parser.add_argument('-ms', '--max_size', default=640, type=int,
                        metavar='N', help='maximum image and video processing size in pixels (default: 640)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--gpus', default=None, nargs='+', type=int, metavar='N',
                        help='list of gpu ids to use (default: all)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='force cpu only')
    parser.add_argument('-v', '--verbose', default=0, type=int,
                        metavar='N', help='print additional information (default: 0)')
    parser.add_argument('-t', '--tmp', default=None, metavar='PATH',
                        help='temporary variable for test')
    args = parser.parse_args()
    main(args.source, args.output, args.det_model, args.lms_model, args.seg_model,
         max_size=args.max_size, gpus=args.gpus, cpu_only=args.cpu_only, verbose=args.verbose)
