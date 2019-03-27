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

parser = argparse.ArgumentParser('concatentae file lists into single files and split it to train/val')
parser.add_argument('-root', help='path to root')

args = parser.parse_args()
if __name__ == '__main__':
    bboxes = [np.load('{}/bboxes_{}.npy'.format(args.root, x)) for x in range(8)]
    bb_cat = np.concatenate(bboxes)
    np.save('{}/bboxes.npy'.format(args.root), bb_cat)

    landmarks = [np.load('{}/landmarks_{}.npy'.format(args.root, x)) for x in range(8)]
    lm_cat = np.concatenate(landmarks)
    np.save('{}/landmarks.npy'.format(args.root), lm_cat)

    eulers = [np.load('{}/eulers_{}.npy'.format(args.root, x)) for x in range(8)]
    eul_cat = np.concatenate(eulers)
    np.save('{}/eulers.npy'.format(args.root), eul_cat)

    frames = [np.load('{}/frames_{}.npy'.format(args.root, x)) for x in range(8)]
    frames_cat = np.concatenate(frames)
    np.save('{}/frames.npy'.format(args.root), frames_cat)

    masks = [np.load('{}/masks_{}.npy'.format(args.root, x)) for x in range(8)]
    masks_cat = np.concatenate(masks)
    np.save('{}/masks.npy'.format(args.root), masks_cat)



