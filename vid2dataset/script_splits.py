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
from itertools import groupby

parser = argparse.ArgumentParser('concatenate file lists into single files and split it to train/val')
parser.add_argument('-root', help='path to root')

args = parser.parse_args()
if __name__ == '__main__':
    min_frames = 4
    ratio = 0.1
    bboxes = np.load("{}/bboxes.npy".format(args.root))
    landmarks = np.load("{}/landmarks.npy".format(args.root))
    eulers = np.load("{}/eulers.npy".format(args.root))
    masks = np.load("{}/masks.npy".format(args.root))
    frames = np.load("{}/frames.npy".format(args.root))

    # Filter images without segmentations
    img_list_names = ["/"+"/".join((f.split('/')[-3:])) for f in frames]
    seg_list_names = ["/"+"/".join((m.split('/')[-3:])) for m in masks]
    img_list_names = np.array(img_list_names)
    masks_list_names = np.array(seg_list_names)

    # Select directories
    keys, groups = [], []
    for key, group in tqdm(groupby(enumerate(img_list_names), lambda x: os.path.split(x[1])[0])):
        keys.append(key)
        group_list = list(group)
        if len(group_list) > min_frames:
            groups.append(group_list)

    # Calculate weights
    weights = np.array([len(g) for g in groups])
    weights = np.sum(weights) / weights
    weights /= np.sum(weights)

    # Generate directory splits
    val_group_indices = np.random.choice(len(groups), int(np.round(len(groups) * ratio)),
                                         replace=False, p=weights).astype(int)
    train_group_indices = np.setdiff1d(np.arange(len(groups)), val_group_indices)

    # Generate splits
    train_indices = np.array([], dtype=int)
    for i in train_group_indices:
        indices, paths = zip(*groups[i])
        train_indices = np.concatenate((train_indices, indices))
    val_indices = np.array([], dtype=int)
    for i in val_group_indices:
        indices, paths = zip(*groups[i])
        val_indices = np.concatenate((val_indices, indices))

    train_indices.sort()
    val_indices.sort()

    train_img_list = img_list_names[train_indices]
    val_img_list = img_list_names[val_indices]

    train_mask_list = masks_list_names[train_indices]
    val_mask_list = masks_list_names[val_indices]


    # Output images splits to file
    train_split_path = os.path.join(args.root, 'images' + '_train.txt')
    val_split_path = os.path.join(args.root, 'images' + '_val.txt')
    np.savetxt(train_split_path, train_img_list, fmt='%s')
    np.savetxt(val_split_path, val_img_list, fmt='%s')

    # Output masks splits to file
    train_split_path = os.path.join(args.root, 'masks' + '_train.txt')
    val_split_path = os.path.join(args.root, 'masks' + '_val.txt')
    np.savetxt(train_split_path, train_mask_list, fmt='%s')
    np.savetxt(val_split_path, val_mask_list, fmt='%s')

    if landmarks is not None:
        train_landmarks = landmarks[train_indices]
        val_landmarks = landmarks[val_indices]
        train_landmarks_path = os.path.join(args.root, 'landmarks' + '_train.npy')
        val_landmarks_path = os.path.join(args.root, 'landmarks' + '_val.npy')
        np.save(train_landmarks_path, train_landmarks)
        np.save(val_landmarks_path, val_landmarks)

    if bboxes is not None:
        train_bboxes = bboxes[train_indices]
        val_bboxes = bboxes[val_indices]
        train_bboxes_path = os.path.join(args.root, 'bboxes' + '_train.npy')
        val_bboxes_path = os.path.join(args.root, 'bboxes' + '_val.npy')
        np.save(train_bboxes_path, train_bboxes)
        np.save(val_bboxes_path, val_bboxes)

    if eulers is not None:
        train_eulers = eulers[train_indices]
        val_eulers = eulers[val_indices]
        train_eulers_path = os.path.join(args.root, 'eulers' + '_train.npy')
        val_eulers_path = os.path.join(args.root, 'eulers' + '_val.npy')
        np.save(train_eulers_path, train_eulers)
        np.save(val_eulers_path, val_eulers)
        pass