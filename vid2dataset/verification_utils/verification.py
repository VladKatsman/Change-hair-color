# Helper function for extracting features from pre-trained models
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from models.model_irse import IR_50
from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def align_(img, crop_size, model_root, filter=False):

    # check if img is PIL
    if type(img) == np.ndarray:
        img = Image.fromarray(img)

    # path to detector models
    op = "{}/onet.npy".format(model_root)
    pp = "{}/pnet.npy".format(model_root)
    rp = "{}/rnet.npy".format(model_root)

    # settings
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    try:  # Handle exception
        _, landmarks = detect_faces(img, ppath=pp, opath=op, rpath=rp)
    except Exception:
        print("Image is discarded due to exception!")
        return 4
    if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
        print("Image is discarded due to non-detected landmarks!")
        return 4
    if filter:
        return True
    else:
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)

        return img_warped


def align_68(img, crop_size, model):
    # check if img is PIL
    bbox = model.face_detector.detect_from_image(img)
    if bbox is None:
        return 4
    landmarks = model.get_landmarks(img, bbox)
    if landmarks is None:
        return 4
    landmarks = landmarks[0]
    left_eye = np.mean(landmarks[36:42, :], axis=0).astype(np.int)
    right_eye = np.mean(landmarks[42:48, :], axis=0).astype(np.int)
    nose = np.mean(landmarks[28:35, :], axis=0).astype(np.int)
    left_mouth = landmarks[48].astype(np.int)
    right_mouth = landmarks[54].astype(np.int)

    # settings
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    facial5points = [left_eye, right_eye, nose, left_mouth, right_mouth]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)

    return img_warped


def align(img, crop_size, img_landmarks):

    landmarks = img_landmarks[0]
    left_eye = np.mean(landmarks[36:42, :], axis=0).astype(np.int)
    right_eye = np.mean(landmarks[42:48, :], axis=0).astype(np.int)
    nose = np.mean(landmarks[28:35, :], axis=0).astype(np.int)
    left_mouth = landmarks[48].astype(np.int)
    right_mouth = landmarks[54].astype(np.int)

    # settings
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    facial5points = [left_eye, right_eye, nose, left_mouth, right_mouth]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)

    return img_warped


def extract_feature(path_img1, path_img2, model_root, input_size=[112, 112]):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # pre-requisites
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)

    # define data loader
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # load model from a checkpoint
    model_path = "{}/backbone_ir50_ms1m_epoch63.pth".format(model_root)
    print("Loading model Checkpoint '{}'".format(model_path))
    model = IR_50(input_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # align and warp images
    input1 = align_(path_img1, input_size[0], model_root)
    input2 = align_(path_img2, input_size[0], model_root)
    if input1 == 4 or input2 == 4:
        return 4

    # transform to torch, norm, extract features
    input1 = transform(input1).unsqueeze(0).to(device)
    output1 = model(input1)
    output1 = l2_norm(output1)

    input2 = transform(input2).unsqueeze(0).to(device)
    output2 = model(input2)
    output2 = l2_norm(output2)

    return output1, output2


def verificate(path_img1, path_img2, model_root, input_size=[112, 112], threshold=1e-2):

    # find features
    feature1, feature2 = extract_feature(path_img1, path_img2, model_root, input_size)

    # find L2 distance between features
    diff = feature1 - feature2
    diff = diff * diff
    pred = torch.lt(diff, threshold)
    num_features = feature1.shape[1]
    confidence = float(pred.sum().float()/num_features)
    print(confidence)

    return confidence


def verificate_pair_features(feature1, feature2, threshold=1e-3):
    # find L2 distance between features
    diff = feature1 - feature2
    diff = diff * diff
    pred = torch.lt(diff, threshold)
    num_features = feature1.shape[1]
    confidence = float(pred.sum().float()/num_features)

    return confidence


class ImageToFeatures:
    def __init__(self, model_root, device, fa_model, input_size=[112, 112]):

        # define data preprocessing
        self.transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # assure that horizontal flip is the same id
        self.insurance = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # initiate face verification model
        self.fa_model = fa_model
        self.model = IR_50(input_size)
        self.model_root = model_root
        model_path = "{}/backbone_ir50_ms1m_epoch63.pth".format(model_root)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

        self.input_size = input_size
        self.device = device

    def __call__(self, image):
        with torch.no_grad():
            # align img
            img_warped = align_68(image, self.input_size[0], self.fa_model)
            if img_warped == 4:
                return 0
            tensor_img_warped = self.transform(img_warped).unsqueeze(0).to(self.device)

            # compute vector of image features
            output = self.model(tensor_img_warped)
            output = l2_norm(output)

        return output


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('FACE VERIFICATION')

    parser.add_argument('-a', '--img1_path', help='path to the first image')
    parser.add_argument('-b', '--img2_path', help='path to the second image')
    parser.add_argument('-p', '--model_path', help='path to the model weights')
    parser.add_argument('-s', '--input_size', default=[112, 112], help='input size of the model')
    parser.add_argument('-t', '--threshold', default=0.001, help='threshold of verification')

    args = parser.parse_args()
    verificate(args.img1_path, args.img2_path, args.model_path, args.input_size, args.threshold)
