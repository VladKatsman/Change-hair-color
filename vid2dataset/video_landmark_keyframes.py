import os
import cv2
from tqdm import tqdm
import numpy as np
import utils
import torch


def main(video_path, out_dir, fa_model, euler_model, ver_model, device, min_size=200,
         frame_sample_ratio=0.1, min_samples=10, sample_limit=500, min_res=720, id_limit=7):
    # cache_file = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0] + '.pkl')
    # cache_file = os.path.splitext(video_path)[0] + '.pkl'

    # Validate video resolution
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read video: ' + video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if width < min_res or height < min_res:
        return
    cap.release()

    # Extract landmarks and bounding boxes
    frame_indexes, landmarks, bboxes, eulers = utils.extract_landmarks_bboxes_from_video(video_path,
                                                                                         fa_model, euler_model,
                                                                                         device=device,
                                                                                         min_size=min_size)
    if len(frame_indexes) == 0:
        return

    # Check if there is only 1 ID per frame
    check = [len(x) == 1 for x in bboxes]
    one_id = sum(check) == len(bboxes)

    if one_id:
        frame_indexes = [np.array(frame_indexes)]
        landmarks = [np.array(landmarks)]
        bboxes = [np.array(bboxes)]
        eulers = [np.array(eulers)]

    else:
        frame_indexes, landmarks, bboxes, eulers = multi_id_processing(video_path, frame_indexes, landmarks,
                                                                              bboxes, eulers, ver_model, id_limit)
    id_num = 0

    for face_id in range(len(frame_indexes)):

        frame_indexes_id = frame_indexes[face_id]
        landmarks_id = landmarks[face_id]
        bboxes_id = bboxes[face_id]
        eulers_id = eulers[face_id]

        # filter by min_samples
        sample_size = min(int(np.round(len(frame_indexes_id) * frame_sample_ratio)), sample_limit)
        if sample_size < min_samples:
            continue
        id_num += 1
        # EULER ANGLES METHOD
        best_sample_indexes = utils.fuse_clusters(eulers_id, 0.3)  # <<- changes from 0.15 Yuval argues for 0.25 min

        if len(best_sample_indexes) > 500:

            # OLD VARIANCE METHOD
            # Normalize landmarks and convert them to the descriptor vectors
            landmark_descs = utils.norm_landmarks(landmarks_id)
            best_sample_indexes = utils.landmarks_var_indexes(frame_indexes_id, landmark_descs, sample_size)
            selected_frame_map = dict(zip(frame_indexes_id[best_sample_indexes], best_sample_indexes))

        else:
            selected_frame_map = dict(zip(frame_indexes_id[best_sample_indexes], best_sample_indexes))

        # Write frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if i in selected_frame_map:
                # Crop frame
                scaled_bbox = utils.scale_bbox(bboxes_id[selected_frame_map[i]], scale=2.0, square=True)
                cropped_frame = utils.crop_img(frame, scaled_bbox)

                # Adjust output landmarks and bounding boxes
                landmarks_id[selected_frame_map[i]] -= scaled_bbox[:2]
                bboxes_id[selected_frame_map[i]][:2] -= scaled_bbox[:2]

                # Write frame to file
                save_path = os.path.join(out_dir, str(id_num))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, 'frame_%04d.jpg' % i), cropped_frame)
        cap.release()

        # Write landmarks and bounding boxes
        np.save(save_path + '_landmarks.npy', landmarks_id[best_sample_indexes])
        np.save(save_path + '_bboxes.npy', bboxes_id[best_sample_indexes])
        np.save(save_path + '_eulers.npy', eulers_id[best_sample_indexes])


def multi_id_processing(video_path, frame_indexes, landmarks, bboxes, eulers, verification_model, id_limit=7):
    """ Iterate each frame and split detected identities information"""

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_frame_features = []
    frame_id = 0
    unique_ids = 0
    structured_frame_indexes = {}
    structured_landmarks = {}
    structured_bboxes = {}
    structured_eulers = {}
    structured_features = {}
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if i in frame_indexes:
            curr_landmarks = landmarks[frame_id]
            curr_bboxes = bboxes[frame_id]
            curr_eulers = eulers[frame_id]
            frame_rgb = frame[:, :, ::-1]
            for detection in range(len(curr_bboxes)):
                scaled_bbox = utils.scale_bbox(curr_bboxes[detection], scale=0.8)
                cropped_frame = utils.crop_img(frame_rgb, scaled_bbox)
                features = verification_model(cropped_frame)
                if type(features) == int:
                    continue
                if frame_id == 0 and unique_ids == 0:
                    last_frame_features.append(features)
                    unique_ids += 1
                    structured_landmarks[0] = [curr_landmarks[detection]]
                    structured_bboxes[0] = [curr_bboxes[detection]]
                    structured_eulers[0] = [curr_eulers[detection]]
                    structured_frame_indexes[0] = [i]
                    structured_features[0] = [features]
                else:
                    conf, index = utils.verificate_frame(last_frame_features, features, threshold=1e-3)
                    if conf > 0.65:
                        structured_landmarks[index].append(curr_landmarks[detection])
                        structured_bboxes[index].append(curr_bboxes[detection])
                        structured_eulers[index].append(curr_eulers[detection])
                        structured_frame_indexes[index].append(i)
                        structured_features[index].append(features)
                        # update last_frame of the subject
                        last_frame_features[index] = features
                    else:
                        # set limitation on number of unique ids, the last one will be dropped out
                        if unique_ids != id_limit:
                            last_frame_features.append(features)
                            unique_ids += 1
                        else:
                            last_frame_features[-1] = features
                        idx = unique_ids - 1
                        structured_landmarks[idx] = [curr_landmarks[detection]]
                        structured_bboxes[idx] = [curr_bboxes[detection]]
                        structured_eulers[idx] = [curr_eulers[detection]]
                        structured_frame_indexes[idx] = [i]
                        structured_features[idx] = [features]
            frame_id += 1
    cap.release()
    # create new arrays of id-related information
    frame_indexes = [[]] * id_limit
    landmarks = [[]] * id_limit
    bboxes = [[]] * id_limit
    eulers = [[]] * id_limit
    features_mean = []

    for key in structured_frame_indexes.keys():
        conf = 0
        feats = torch.cat(structured_features[key])
        mean_feature = torch.mean(feats, 0).unsqueeze(0)

        # for second identity do  verification of similarity
        if key != 0:
            conf, index = utils.verificate_frame(features_mean, mean_feature, threshold=1e-3)

        if conf > 0.7:  # 0.8 was working
            landmarks[index] = np.concatenate([landmarks[index], np.array(structured_landmarks[key])])
            bboxes[index] = np.concatenate([bboxes[index], np.array(structured_bboxes[key])])
            eulers[index] = np.concatenate([eulers[index], np.array(structured_eulers[key])])
            frame_indexes[index] = np.concatenate([frame_indexes[index], np.array(structured_frame_indexes[key])])
        else:
            frame_indexes[key] = np.array(structured_frame_indexes[key])
            landmarks[key] = np.array(structured_landmarks[key])
            bboxes[key] = np.array(structured_bboxes[key])
            eulers[key] = np.array(structured_eulers[key])
            features_mean.append(mean_feature)

    return frame_indexes, landmarks, bboxes, eulers


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_keyframes')
    parser.add_argument('video_path', help='path to video file')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    parser.add_argument('-r', '--models_root', metavar='DIR', help='root dir for alignment models')
    parser.add_argument('-fa', '--face_alignment_model', metavar='DIR', help='face_alignment model')
    parser.add_argument('-em', '--euler_angles_model', metavar='DIR', help='euler angles model')
    parser.add_argument('-ver', '--verification_model', metavar='DIR', help='verification model')

    parser.add_argument('-mb', '--min_bbox_size', default=200, type=int, metavar='N',
                        help='minimum bounding box size')
    parser.add_argument('-fs', '--frame_samples', default=0.1, type=float, metavar='F',
                        help='the number of samples per frame')
    parser.add_argument('-ms', '--min_samples', default=5, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-sl', '--sample_limit', default=100, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-mr', '--min_res', default=720, type=int, metavar='N',
                        help='minimum video resolution (height pixels)')
    args = parser.parse_args()
    main(args.video_path, args.output, args.models_root, args.face_alignment_model, args.euler_angles_model, args.verification_model,
         args.min_bbox_size, args.frame_samples, args.min_samples, args.sample_limit, args.min_res)

