import video_landmark_keyframes
import os
import traceback
import logging
import face_alignment
import utils
import torch
from glob import glob
from models import hopenet
from verification_utils import verification


def main(txtfile, out_dir, models_root, min_size=120, frame_sample_ratio=0.1, min_samples=5, sample_limit=100,
         min_res=720, id_limit=7, cudev=0):

    # vid_paths = glob(os.path.join(in_dir, '*.mp4'))  <-- in_dir replaced with vid_paths
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudev) #  <-- added in order to use multiple gpus to process IJB-C

    vid_paths = []
    with open(txtfile) as a:
        for line in a:
            vid_paths.append(line.rstrip())



    # INITIALIZE MODELS
    device, gpus = utils.set_device()


    # Initialize detection and landmarks extraction
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # Initialize eulers angles model
    Gp = hopenet.Hopenet()
    Gp.to(device)
    path2w = "{}/hopenet_robust_alpha1.pkl".format(models_root)
    weights = torch.load(path2w)
    Gp.load_state_dict(weights)
    Gp.eval()

    # Initialize verification model
    verificator = verification.ImageToFeatures(models_root, device, fa)

    # For each video file
    for vid_path in sorted(vid_paths):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        curr_out_dir = os.path.join(out_dir, vid_name)

        if os.path.exists(curr_out_dir):
            print('Skipping "%s"' % vid_name)
            continue
        else:
            print('Processing "%s"...' % vid_name)
            # os.mkdir(curr_out_dir)

        # Process video
        try:
            video_landmark_keyframes.main(vid_path, curr_out_dir, fa, Gp, verificator, device,
                                          min_size, frame_sample_ratio, min_samples, sample_limit, min_res, id_limit)
        except Exception as e:
            logging.error(traceback.format_exc())


    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('video_landmarks_keyframes_batch')
    parser.add_argument('input', metavar='DIR', help='input directory')
    parser.add_argument('-m', '--models_root', metavar='DIR', help='model root directory')
    parser.add_argument('-o', '--output', metavar='DIR', help='output directory')
    parser.add_argument('-mb', '--min_bbox_size', default=200, type=int, metavar='N',
                        help='minimum bounding box size')
    parser.add_argument('-fs', '--frame_samples', default=0.1, type=float, metavar='F',
                        help='the number of samples per video')
    parser.add_argument('-ms', '--min_samples', default=5, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-sl', '--sample_limit', default=100, type=int, metavar='N',
                        help='the limit on the number of samples')
    parser.add_argument('-mr', '--min_res', default=720, type=int, metavar='N',
                        help='minimum video resolution (height pixels)')
    parser.add_argument('-il', '--id_limit', default=7, type=int, metavar='N',
                        help='the limit on the number of identities')
    args = parser.parse_args()
    main(args.input, args.output, args.models_root, args.min_bbox_size, args.frame_samples, args.min_samples, args.sample_limit,
         args.min_res, args.id_limit)

