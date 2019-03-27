import os
import shutil
import torch
import random
import torch.nn.init as init
import numpy as np
import warnings
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F


def init_weights(m, init_type='kaiming', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """ Saves a model's checkpoint.
    :param exp_dir: Experiment directory to save the checkpoint into.
    :param base_name: The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
    :param state: The model state to save.
    :param is_best: If True <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + '_latest.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + '_best.pth'))


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


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


def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred    = pred.t()
    pred = pred.view(batch_size, -1)
    target.view(-1, 1).expand_as(pred)

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:, :k].view(-1).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def txt2list(path):
    """ Reads .txt file and saves it as list"""
    l = []
    with open(path) as a:
        for line in a:
            l.append(line.rstrip())
    return l


def load_dict_from_file(file):
    """ wrapper to load dict files saved as .txt"""
    f = open(file, 'r')
    data = f.read()
    f.close()
    return eval(data)


def build_distribution(root):
    """ builds dictionary of classes distribution"""
    path_to_attributes = os.path.join(root, "new_attr_list.txt")
    path_to_dic_attributes = os.path.join(root, "new_attr_dic.txt")
    attributes_dic = load_dict_from_file(path_to_dic_attributes)
    lst = open(path_to_attributes).read().splitlines()
    hist = [0] * len(attributes_dic)
    for line in lst:
        attributes = line.rstrip().split(',')[1:]
        attributes = [1 if int(x) == 1 else 0 for x in attributes]
        for i in range(len(attributes)):
            hist[i] += attributes[i]
    return hist


def class_weights(root):
    """ using class distribution to create list with weights in order to balance dataset"""
    hist = build_distribution(root)
    weights = [1/x for x in hist]
    return weights


def weights_to_sample(root):
    """ computes weights for each sample in the multilabel attributes train dataset """
    path_to_attributes = os.path.join(root, "new_attr_list.txt")
    w = class_weights(root)
    W = []
    lst = open(path_to_attributes).read().splitlines()
    for line in lst:
        attributes = line.rstrip().split(',')[1:]
        attributes = [1 if int(x) == 1 else 0 for x in attributes]
        w_max = np.max(np.array(attributes) * np.array(w))
        W.append(w_max)

    w_train = []
    path_to_train = os.path.join(root, "train.txt")
    lst = open(path_to_train).read().splitlines()
    for line in lst:
        index = int(line.rstrip().split('.')[0]) - 1
        w_train.append(W[index])

    return w_train


class RunningScore(object):
    """ Alternative for average meter for multiclass classification problem"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.acc_cls = [0] * n_classes
        self.tp = [0] * n_classes
        self.tn = [0] * n_classes
        self.num_pos = [0] * n_classes
        self.num_neg = [0] * n_classes

    def update(self, label_trues, label_preds):
        self.get_acc(label_trues, label_preds)

    def get_acc(self, true, pred):
        """Returns accuracy score evaluation result"""
        with torch.no_grad():
            eps = 1e-6
            batch_size = true.size(0)
            pred = torch.ge(pred, 0.5)
            pred = pred.view(1, -1)
            num_pos = true.byte().view(1, -1)[0]
            num_neg = true.byte().eq(0.0).view(1, -1)[0]
            mask_t = torch.full_like(pred, 2)
            pos_pred = torch.where(pred == 1, pred, mask_t)
            neg_pred = torch.where(pred == 0, pred, mask_t)
            tp = pos_pred[0].eq(num_pos)
            tn = neg_pred[0].eq(num_pos)
            tp = tp.squeeze(dim=0)
            tn = tn.squeeze(dim=0)
            for n in range(self.n_classes):
                i = n * batch_size
                j = i + batch_size
                self.tp[n] += tp[i:j].sum().item()
                self.tn[n] += tn[i:j].sum().item()
                self.num_pos[n] += num_pos[i:j].sum().item()
                self.num_neg[n] += num_neg[i:j].sum().item()
                self.acc_cls[n] = 0.5 * ((self.tp[n] + eps) / (self.num_pos[n] + eps) + (self.tn[n] +eps) / (self.num_neg[n] + eps))

    def reset(self):
        self.acc_cls = [0] * self.n_classes
        self.tp = [0] * self.n_classes
        self.tn = [0] * self.n_classes
        self.num_pos = [0] * self.n_classes
        self.num_neg = [0] * self.n_classes

# TMP in order not to change too much production code


def get_main_bbox(bboxes, img_size):
    if len(bboxes) == 0:
        return None

    # Calculate frame max distance and size
    img_center = np.array([img_size[1], img_size[0]]) * 0.5
    max_dist = 0.25 * np.linalg.norm(img_size)
    max_size = 0.25 * (img_size[0] + img_size[1])

    # For each bounding box
    scores = []
    for bbox in bboxes:
        # Calculate center distance
        bbox_center = bbox[:2] + bbox[2:] * 0.5
        bbox_dist = np.linalg.norm(bbox_center - img_center)

        # Calculate bbox size
        bbox_size = bbox[2:].mean()

        # Calculate central ratio
        central_ratio = 1.0 if max_size < 1e-6 else (1.0 - bbox_dist / max_dist)
        central_ratio = np.clip(central_ratio, 0.0, 1.0)

        # Calculate size ratio
        size_ratio = 1.0 if max_size < 1e-6 else (bbox_size / max_size)
        size_ratio = np.clip(size_ratio, 0.0, 1.0)

        # Add score
        score = (central_ratio + size_ratio) * 0.5
        scores.append(score)

    return bboxes[np.argmax(scores)]


def rgb2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(o) for o in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)


def bgr2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(o, normalize) for o in img]
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)


def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor2rgb(img_tensor):
    output_img = unnormalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img * 255).astype('uint8')

    return output_img


def tensor2bgr(img_tensor):
    output_img = tensor2rgb(img_tensor)
    output_img = output_img[:, :, ::-1]

    return output_img


def my_collate(batch):
    input = [item[0] for item in batch]
    target = [item[1] for item in batch]
    images = [item[2] for item in batch]
    input = torch.stack(input, 0)
    return [input, target, images]


def main():
    import torch

    output = torch.rand(2, 10, 1, 1)
    target = torch.LongTensor(range(2))
    acc = topk_accuracy(output, target, topk=(1, 5))
    print(acc)


if __name__ == "__main__":
    main()
