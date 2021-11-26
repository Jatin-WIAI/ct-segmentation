import torch
import torch.nn as nn
import torchvision.models.segmentation as tvsegmentation
from segmentation_models_pytorch import (FPN, PAN, DeepLabV3, DeepLabV3Plus,
                                         Linknet, MAnet, PSPNet, Unet,
                                         UnetPlusPlus)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class deeplabv3_resnet50(nn.Module):
    def __init__(self, num_classes, basemodel=tvsegmentation.deeplabv3_resnet50, pretrained=True):
        super(deeplabv3_resnet50, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = DeepLabHead(2048, num_classes)
        # self.basemodel.classifier[4] = nn.Conv2d(
        #     in_channels=256,
        #     out_channels=num_classes,
        #     kernel_size=1,
        #     stride=1
        # )

    def forward(self, x):
        x = self.basemodel(x)
        return x

class deeplabv3_resnet101(nn.Module):
    def __init__(self, num_classes, basemodel=tvsegmentation.deeplabv3_resnet101, pretrained=True):
        super(deeplabv3_resnet101, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class deeplabv3_mobilenet_v3_large(nn.Module):
    def __init__(self, num_classes, basemodel=tvsegmentation.deeplabv3_mobilenet_v3_large, pretrained=True):
        super(deeplabv3_mobilenet_v3_large, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        x = self.basemodel(x)
        return x
