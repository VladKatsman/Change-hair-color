import os
from video_landmark_keyframes_batch import main
import argparse

parser = argparse.ArgumentParser('video_landmarks_keyframes')
parser.add_argument('-videos', help='path to txt file with videos splits')
parser.add_argument('-gpu_num', help='gpu num')

args = parser.parse_args()
if __name__ == '__main__':

    input = args.videos
    models_root = '/data/dev/models'
    output = '/data/datasets/output'
    min_bbox_size = 200
    frame_samples = 0.1
    min_samples = 10
    sample_limit = 500
    min_res = 720
    id_limit = 7
    cudev = args.gpu_num

    main(input, output, models_root, min_bbox_size, frame_samples, min_samples, sample_limit, min_res, id_limit, cudev)

os.system('sudo shutdown')
