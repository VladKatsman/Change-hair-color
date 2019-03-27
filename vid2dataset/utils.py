import os
import cv2
import pickle
import numpy as np
import torch
import random
import torchvision.transforms.functional as F
import cv2 as cv
from torchvision import transforms
from face_manipulations.utils.obj_factory import obj_factory
from tqdm import tqdm
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances
from verification_utils import verification


def extract_landmarks_bboxes_from_video(video_path, fa, eulers_model, device, min_size, cache_file=None):
    """ Extract face landmarks and bounding boxes from video and also read / write them from cache file.
    :param video_path: Path to video file.
    :param cache_file: Path to file to save the landmarks and bounding boxes in.
        By default it is saved in the same directory of the video file with the same name and extension .pkl.
    :param fa: face alignment model to detect faces (bboxes and landmarks)
    :param euler: euler angles model to find Yaw, Pitch and Roll

    :return: tuple (numpy.array, numpy.array, numpy.array):
        frame_indices: The frame indices where a face was detected.
        landmarks: Face landmarks per detected face in each frame.
        bboxes: Bounding box per detected face in each frame.
    """
    cache_file = os.path.splitext(video_path)[0] + '.pkl' if cache_file is None else cache_file
    if not os.path.exists(cache_file):
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError('Failed to read video: ' + video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        torch.set_grad_enabled(False)

        # Initialize frame information containers
        frame_indices = []
        landmarks = []
        bboxes = []
        euler = []

        # For each frame in the video
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if frame is None:
                continue
            frame_rgb = frame[:, :, ::-1]
            detected_faces = fa.face_detector.detect_from_image(frame.copy())
            if len(detected_faces) == 0:
                continue

            # tmp containers for multidetection
            frame_landmarks = []
            frame_bboxes = []
            frame_eulers = []

            for j in range(len(detected_faces)):
                curr_bbox = detected_faces[j]

                # filter detections according to the bboxes threshold
                if (curr_bbox[2] - curr_bbox[0] > min_size) or (curr_bbox[3] - curr_bbox[1] > min_size):

                    # landmarks and bboxes
                    curr_bbox_input = [curr_bbox[:]]
                    preds = fa.get_landmarks(frame_rgb, curr_bbox_input)
                    if len(preds) == 0:
                        continue
                    curr_landmarks = preds[0]
                    curr_bbox = curr_bbox[:4]

                    # Convert bounding boxes format from [min, max] to [min, size]
                    curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1

                    # Calculate euler angles
                    eulers = compute_euler_angles(curr_bbox, frame_rgb, eulers_model, device)

                    # Append to tmp list
                    frame_landmarks.append(curr_landmarks)
                    frame_bboxes.append(curr_bbox)
                    frame_eulers.append(eulers)

            # Append to the main list
            frame_indices.append(i)
            landmarks.append(frame_landmarks)
            bboxes.append(frame_bboxes)
            euler.append(frame_eulers)

        # Save landmarks and bounding boxes to file
        with open(cache_file, "wb") as fp:  # Pickling
            pickle.dump(frame_indices, fp)
            pickle.dump(landmarks, fp)
            pickle.dump(bboxes, fp)
            pickle.dump(euler, fp)
    else:
        # Load landmarks and bounding boxes from file
        with open(cache_file, "rb") as fp:  # Unpickling
            frame_indices = pickle.load(fp)
            landmarks = pickle.load(fp)
            bboxes = pickle.load(fp)
            euler = pickle.load(fp)

    return frame_indices, landmarks, bboxes, euler

                                            ######################
                                            ###      MISC      ###
                                            ######################


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


def rgb2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def norm_landmarks(landmarks):
    """ Normalize landmarks and convert them to descriptor vectors
    """

    landmark_descs = landmarks.copy()
    for lms in landmark_descs:
        lms -= np.mean(lms, axis=0)
        lms /= (np.max(lms) - np.min(lms))
    landmark_descs = landmark_descs.reshape(landmark_descs.shape[0], -1)  # Reshape landmarks to vectors
    landmark_descs -= np.mean(landmark_descs, axis=0)  # Normalize landmarks

    return landmark_descs


def uniform_sample_with_min_dist(limit, n, min_dist):
    slack = limit - 1 - min_dist * (n - 1)
    steps = random.randint(0, slack)

    increments = np.hstack([np.ones((steps,)), np.zeros((n,))])
    np.random.shuffle(increments)

    locs = np.argwhere(increments == 0).flatten()
    samples = np.cumsum(increments)[locs] + min_dist * np.arange(0, n)

    return np.array(samples, dtype=int)


def crop_img(img, bbox):
    min_xy = bbox[:2]
    max_xy = bbox[:2] + bbox[2:] - 1
    min_xy[0] = min_xy[0] if min_xy[0] >= 0 else 0
    min_xy[1] = min_xy[1] if min_xy[1] >= 0 else 0
    max_xy[0] = max_xy[0] if max_xy[0] < img.shape[1] else (img.shape[1] - 1)
    max_xy[1] = max_xy[1] if max_xy[1] < img.shape[0] else (img.shape[0] - 1)

    return img[min_xy[1]:max_xy[1] + 1, min_xy[0]:max_xy[0] + 1]


def scale_bbox(bbox, scale=1.35, square=True):
    bbox_center = bbox[:2] + bbox[2:] / 2
    bbox_size = np.round(bbox[2:] * scale).astype(int)
    if square:
        bbox_max_size = np.max(bbox_size)
        bbox_size = np.array([bbox_max_size, bbox_max_size], dtype=int)
    bbox_min = np.round(bbox_center - bbox_size / 2).astype(int)
    bbox_scaled = np.concatenate((bbox_min, bbox_size))

    return bbox_scaled


def landmarks_var_indexes(frame_indexes, landmark_descs, sample_size, iters=20000):
    """ Finds best frame distribution using variance and landmarks

    :param landmark_descs: normalized landmarks transformed into descriptor vectors
    :param frame_indexes: indexes to choose from
    :param sample_size: number of samples to use
    :param iters: number of iterations of the method

    :return: best sample indexes
    """
    max_mean_dist = 0.
    best_sample_indexes = None
    for i in range(iters):
        sample_indexes = uniform_sample_with_min_dist(len(frame_indexes), sample_size, 4)
        landmark_desc_samples = landmark_descs[sample_indexes]
        dist = euclidean_distances(landmark_desc_samples, landmark_desc_samples)
        mean_dist = np.mean(dist)
        if mean_dist > max_mean_dist:
            max_mean_dist = mean_dist
            best_sample_indexes = sample_indexes

    return best_sample_indexes


def fuse_clusters(points, r=0.5):
    """ Finds best distribution of frames depending on parameter r

    :param points: The points to cluster
    :param r: The radius for which to fuse the points
    :return: indices of remaining points.
    """
    points = points[:, :2]  # drop roll dimension
    kdt = cKDTree(points)
    indices = kdt.query_ball_point(points, r=r)

    # Build sorted neightbor list
    neighbors = [(i, l) for i, l in enumerate(indices)]
    neighbors.sort(key=lambda t: len(t[1]), reverse=True)

    # Mark remaining indices
    keep = np.ones(points.shape[0], dtype=bool)
    for i, cluster in neighbors:
        if not keep[i]:
            continue
        for j in cluster:
            if i == j:
                continue
            keep[j] = False

    return np.nonzero(keep)[0]


def filter_frames(video_path, model_root, frame_indexes, bboxes, crop_size=112):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    id = 0
    filtered_indexes = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if i in frame_indexes:
            if frame is None:
                continue
            bbox = bboxes[id]
            frame_rgb = frame[:, :, ::-1]
            image = crop_img(frame_rgb, bbox)
            out = verification.align_(image, crop_size, model_root, filter=True)
            if out:
                filtered_indexes.append(id)
            id += 1
    cap.release()

    return filtered_indexes


def decrop_mask(original_mask, cropped_mask, bbox, changes):
    """ Removes all changes being made by crop_img and scale_bbox functions

    :param img: img which changes have to be removed
    :param changes: padding to be removed, output of the "crop_img" function
    :param bbox: bboxes being used to crop an image (output of "scale_bbox" function)

    :return: original Segmentation Mask with background, hair and face classes
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
                                        #################################
                                        ###      MODELs WRAPPERs      ###
                                        #################################


def face_segmentation_filter(cropped_frame, seg_model_path, model_in_size=256, threshold=0.15):
    """ Checks if cropped frame includes face fragment about desired threshold

    :param cropped_frame: frame cropped using scaled bboxes of face detector model
    :param model_in_size: what input size required for the seg model. Set to None if not required
    :param threshold: threshold of predictions to filter face

    :return: True, if face is accepted, False otherwise
    """
    device, gpus = set_device()

    # Load face segmentation model
    print('Loading face segmentation model: "' + os.path.basename(seg_model_path) + '"...')
    if seg_model_path.endswith('.pth'):
        checkpoint = torch.load(seg_model_path)
        segmentation_net = obj_factory(checkpoint['arch']).to(device)
        segmentation_net.load_state_dict(checkpoint['state_dict'])
    else:
        segmentation_net = torch.jit.load(seg_model_path, map_location=device)
    if segmentation_net is None:
        raise RuntimeError('Failed to load face segmentation model!')

    # switch to evaluation mode
    segmentation_net.eval()
    torch.set_grad_enabled(False)

    # preprocessing
    if model_in_size is not None:
        source = cv2.resize(cropped_frame, (model_in_size, model_in_size), interpolation=cv2.INTER_CUBIC)
    source = rgb2tensor(source, normalize=True)
    source = source.to(device)
    preds = segmentation_net(source)
    preds = preds.max(1)[1].cpu().numpy().astype(np.uint8)
    preds[preds == 2] = 0


def verificate_frame(face_id_features, face_id_feature, threshold=1e-3):
    """ Find distances between one face object on the current frame and unique face objects on the previous frames

    :param face_id_features: list of face features from previous frames
    :param face_id_feature: one of the face features of the current frame

    :return: confidence of face id to belong to one of the previous ids, index of the face id
    """
    if len(face_id_features) > 1:
        next_frame_features = face_id_feature.expand(len(face_id_features), face_id_feature.shape[1])
        previous_frames_features = torch.stack(face_id_features).squeeze(1)
    elif len(face_id_features) == 1:
        next_frame_features = face_id_feature
        previous_frames_features = face_id_features[0]
    else:
        return 0, 0
    diff = next_frame_features - previous_frames_features
    diff = diff * diff
    pred = torch.lt(diff, threshold)  # threshold for verification model
    confidence = pred.sum(-1).float() / 512  # 512 is num of features of the ver model
    conf, index = confidence.max(0)
    conf, index = float(conf), int(index)

    return conf, index


def compute_euler_angles(bbox, image, euler_model, device):
    """ Computes euler angles using pretrained hopenet model: https://github.com/natanielruiz/deep-head-pose

    :param bbox: bbox of the face
    :param image: image with face situated at bbox region
    :param euler_model: pretrained hopenet model

    :return: tuple of (yaw, pitch, roll)
    """
    curr_bbox = bbox.copy()
    frame_rgb = image.copy()
    scaled_bbox = scale_bbox(curr_bbox, scale=2.0, square=True)
    cropped_frame = crop_img(frame_rgb, scaled_bbox)
    cropped_frame = Image.fromarray(cropped_frame)
    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                               std=[0.5, 0.5, 0.5])])
    euler_tensor = transformations(cropped_frame)
    euler_tensor = euler_tensor.unsqueeze(0).to(device)
    yaw, pitch, roll = euler_model(euler_tensor)

    return [float(yaw), float(pitch), float(roll)]
