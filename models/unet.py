import torch
import torch.nn as nn
import torch.nn.functional as F

from face_tran.utils import utils


pretrained_models = {
    'face_seg_256': '/data/dev/projects/face_tran_experiments/results/unet_faces_256/model_best.pth',
    'face_seg_512': '/data/dev/projects/face_tran_experiments/results/unet_faces_512/model_best.pth',
    'pascal': '/data/dev/projects/face_tran_experiments/results/pascal12_unet/model_best.pth',
}

# Code is borrowed from
# https://github.com/meetshah1995/pytorch-semseg/commits/master/ptsemseg/models/unet.py


class UNet(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=21,
        in_channels=3,
        is_batchnorm=True,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3])
        self.up_concat3 = UnetUp(filters[3], filters[2])
        self.up_concat2 = UnetUp(filters[2], filters[1])
        self.up_concat1 = UnetUp(filters[1], filters[0])

        # weight init
        for m in self.modules():
            utils.init_weights(m)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2(in_size, out_size, False)

        self.conv1d = nn.Conv1d(in_size, out_size, kernel_size=(1,1))

    def forward(self, inputs1, inputs2):
        outputs2 = F.interpolate(inputs2, scale_factor=2, mode='bilinear')
        outputs2 = self.conv1d(outputs2,)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


def unet(num_classes=21, feature_scale=1, is_batchnorm=True, pretrained=False):
    """ Builds UNet model

    :param is_deconv: by default False, meaning Bilinear2d, True to change it to ConvTranspose2d
    :param feature_scale: scales number of parameters in the network for your needs (by default 1 is no scale)
    :param pretrained: You can choose pretrained if you want to use the model pretrained on PASCAL VOC
    """
    if pretrained:
        model_path = pretrained_models['pascal']
        model = UNet(n_classes=num_classes, feature_scale=feature_scale, is_batchnorm=is_batchnorm)
        checkpoint = torch.load(model_path)
        weights = checkpoint['state_dict']
        weights['notinuse'] = weights.pop('final.weight')
        weights['notinuse2'] = weights.pop('final.bias')
        model.load_state_dict(weights, strict=False)
    else:
        model = UNet(n_classes=num_classes, feature_scale=feature_scale, is_batchnorm=is_batchnorm)

    return model
