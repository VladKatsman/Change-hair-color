import os
import argparse
import utils
import torch
import numpy as np
import cv2 as cv
from glob import glob
from models import unet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser('video_landmarks_keyframes')
parser.add_argument('-videos', help='path to txt file with videos splits')
parser.add_argument('-gpu_num', help='gpu num')
parser.add_argument('-path_to_model', help='path to model')
parser.add_argument('-root', help='path to root')

args = parser.parse_args()
if __name__ == '__main__':

    models_root = '/data/dev/models'
    cudev = args.gpu_num
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])])

    # set model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device, gpus = utils.set_device()
    checkpoint = torch.load(args.path_to_model)
    model = unet.unet(3, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # find paths
    vid_paths = []
    with open(args.videos) as a:
        for line in a:
            path = line.rstrip()
            vid_paths.append(path)
    frames_list = []
    masks_list = []
    bboxes_list = []
    landmarks_list = []
    eulers_list = []

    for path in tqdm(vid_paths):
        if os.path.exists(path):
            folder = os.path.split(path)[-1]
            ids = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
            for id_n in ids:
                frames = glob("{}/{}/*".format(path, id_n))
                frames.sort(reverse=True)
                landmarks = np.load("{}/{}_landmarks.npy".format(path, id_n))
                bboxes = np.load("{}/{}_bboxes.npy".format(path, id_n))
                eulers = np.load("{}/{}_eulers.npy".format(path, id_n))
                idx = len(bboxes) - 1
                for frame in frames:
                    frame_rgb = cv.imread(frame)[:, :, ::-1]
                    h, w, _ = frame_rgb.shape
                    original_mask = np.zeros((h, w))
                    curr_bbox = bboxes[idx]
                    scaled_bbox = utils.scale_bbox(curr_bbox, scale=2.0, square=True)
                    cropped_frame, changes = utils.crop_img_with_padding(frame_rgb, scaled_bbox)
                    h, w, _ = cropped_frame.shape
                    scaled_frame = cv.resize(cropped_frame, (256, 256), cv.INTER_CUBIC)
                    tensor = transformations(scaled_frame).unsqueeze(0).to(device)
                    preds = model(tensor)
                    preds = preds.max(1)[1].cpu().numpy().astype(np.uint8).squeeze(0)

                    # workaround against fake face label around hair region
                    face_pred = preds.copy()
                    face_pred[face_pred == 2] = 0
                    # kernel = np.ones((5, 5), np.uint8)
                    # face_pred = cv.erode(face_pred, kernel, iterations=1)
                    face_pred[preds == 2] = 2

                    # upsampling is not robust with open cv, using PIL.Image
                    pil = Image.fromarray(face_pred)
                    scaled_frame = pil.resize((h, w))
                    scaled_frame = np.array(scaled_frame)  # back to numpy
                    try:
                        face_pred = utils.decrop_mask(original_mask, scaled_frame, scaled_bbox, changes)
                    except ValueError:
                        idx -= 1
                        continue
                    a = np.count_nonzero(face_pred) / face_pred.size
                    if a > 0.15:
                        if ((face_pred == 2).sum()/((face_pred == 0) + (face_pred == 1)).sum()) < 0.05:
                            face_pred[face_pred == 2] = 0

                        mask = Image.fromarray(face_pred.astype(np.uint8))
                        if len(np.unique(face_pred)) == 3:
                            imP = mask.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=3)
                            reverse_colors = np.array(imP)
                            reverse_colors[reverse_colors == 0] = 3
                            reverse_colors[reverse_colors == 2] = 0
                            reverse_colors[reverse_colors == 3] = 2
                            imP = Image.fromarray(reverse_colors, mode='P')
                            imP.putpalette([
                                0, 0, 0,  # index 0 is black (background)
                                0, 255, 0,  # index 1 is green (face)
                                255, 0, 0,  # index 2 is red (hair)
                            ])
                        else:
                            imP = mask.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=2)
                            reverse_colors = np.array(imP)
                            reverse_colors[reverse_colors == 1] = 3
                            reverse_colors[reverse_colors == 0] = 1
                            reverse_colors[reverse_colors == 3] = 0
                            imP = Image.fromarray(reverse_colors, mode='P')
                            imP.putpalette([
                                0, 0, 0,  # index 0 is black (background)
                                0, 255, 0,  # index 1 is green (face)
                            ])
                        i_path = "{}/images/{}_{}".format(args.root, folder, id_n)
                        m_path = "{}/masks/{}_{}".format(args.root, folder, id_n)
                        if not os.path.exists(i_path):
                            os.makedirs(i_path)
                        if not os.path.exists(m_path):
                            os.makedirs(m_path)

                        frames_list.append("{}/images/{}_{}/{}.jpg".format(args.root, folder, id_n, idx))
                        masks_list.append("{}/masks/{}_{}/{}.png".format(args.root, folder, id_n, idx))
                        bboxes_list.append(curr_bbox)
                        landmarks_list.append(landmarks[idx])
                        eulers_list.append(eulers[idx])
                        imP.save("{}/masks/{}_{}/{}.png".format(args.root, folder, id_n, idx))
                        Image.fromarray(frame_rgb).save("{}/images/{}_{}/{}.jpg".format(args.root, folder, id_n, idx))
                    idx -= 1

    np.save('{}/frames_{}'.format(args.root, cudev), frames_list)
    np.save('{}/masks_{}'.format(args.root, cudev), masks_list)
    np.save('{}/bboxes_{}'.format(args.root, cudev), bboxes_list)
    np.save('{}/landmarks_{}'.format(args.root, cudev), landmarks_list)
    np.save('{}/eulers_{}'.format(args.root, cudev), eulers_list)
