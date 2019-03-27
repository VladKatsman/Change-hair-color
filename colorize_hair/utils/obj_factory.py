import os
import importlib


known_modules = {
    # criterions
    'ssim': 'face_tran.criterions.ssim',

    # data
    'domain_dataset': 'face_tran.data.domain_dataset',
    'generic_face_dataset': 'face_tran.data.generic_face_dataset',
    'image_list_dataset': 'face_tran.data.image_list_dataset',
    'face_list_dataset': 'face_tran.data.face_list_dataset',
    'pascal_list_dataset': 'face_tran.data.pascal_list_dataset',  # added by Vlad for PASCAL VOC 2012
    'face_seg_dataset': 'face_tran.data.face_seg_dataset',  # added by Vlad for "FACE" dataset
    'pascal_unseen_dataset': 'face_tran.data.pascal_unseen_dataset', # added by Vlad for FineTune ResUNet
    'hair_face_background_dataset': 'face_tran.data.hair_face_background_dataset',  # added by for face/hair dataset
    'celeba_dataset': 'face_tran.data.celeba_dataset',  # added by Vlad for celeba attributes classification
    'celebaHQ_dataset': 'face_tran.data.celebaHQ_dataset',  # added by Vlad for celebaHQ classification
    'vggface2_dataset': 'face_tran.data.vggface2_dataset',  # added by Vlad for vggface2 segmentation with UNet
    'FacePairLandmarksDataset': 'face_tran.data.face_landmarks_dataset',  # landmarks for ReeGAN
    'celeba_benchmark_dataset': 'face_tran.data.celeba_benchmark_dataset',  # benchmark dataset
    'lfw_figaro_dataset': 'face_tran.data.lfw_figaro_dataset',        # lfw + figaro dataset

    # models
    'pg_clipped_enc_dec': 'face_tran.models.pg_clipped_enc_dec',
    'pg_sep_unet': 'face_tran.models.pg_sep_unet',
    'pg_enc_dec': 'face_tran.models.pg_enc_dec',
    'resnet': 'face_tran.models.resnet',
    'classifiers': 'face_tran.models.classifiers',
    'decoders': 'face_tran.models.decoders',
    'discriminators': 'face_tran.models.discriminators',
    'vgg': 'face_tran.models.vgg',  # added by Vlad
    'fcn': 'face_tran.models.fcn',  # added by Vlad
    'unet': 'face_tran.models.unet',  # added by Vlad
    'unet_res': 'face_tran.models.unet_res',  # added by Vlad (Yuval implementation)
    'pspnet': 'face_tran.models.pspnet',  # added by Vlad
    'refinenet': 'face_tran.models.refinenet',  # added by Vlad
    'res_unet_split': 'face_tran.models.res_unet_split',
    # utils
    # added by Vlad for segmentation
    'seg_transforms': 'face_tran.utils.seg_transforms',
    'losses': 'face_tran.utils.losses',
    'schedulers': 'face_tran.utils.schedulers',
    'landmark_transforms': 'face_tran.utils.landmark_transforms',

    # Torch
    'nn': 'torch.nn',
    'optim': 'torch.optim',
    'lr_scheduler': 'torch.optim.lr_scheduler',


    # Torchvision
    'datasets': 'torchvision.datasets',
    'transforms': 'torchvision.transforms'
}

known_classes = {
}


def extract_args(*args, **kwargs):
    return args, kwargs


def obj_factory(obj_exp, *args, **kwargs):
    if not isinstance(obj_exp, str):
        return obj_exp

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(known_modules[module_name] if module_name in known_modules else module_name)
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def main(obj_exp):
    obj = obj_factory(obj_exp)
    print(obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('utils test')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)
