# This files contain model definition for CNN classification networks

import torchvision.models as tvmodels
import torch.nn as nn


class inception_v3(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.inception_v3, pretrained=True):
        super(inception_v3, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class mobilenet_v2(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.mobilenet_v2, pretrained=True):
        super(mobilenet_v2, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class mobilenet_v3_large(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.mobilenet_v3_large, pretrained=True):
        super(mobilenet_v3_large, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[3] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class squeezenet1_0(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.squeezenet1_0, pretrained=True):
        super(squeezenet1_0, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.basemodel(x)
        return x


class googlenet(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.googlenet, pretrained=True):
        super(googlenet, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class vgg16(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.vgg16, pretrained=True):
        super(vgg16, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[6] = nn.Linear(
            in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class vgg16_bn(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.vgg16_bn, pretrained=True):
        super(vgg16_bn, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[6] = nn.Linear(
            in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x
# For vgg we make a class vgg which reduces the number of classifier layers. For resnet, it is not needed as just calling the model with num_classes gives
# the same model. The same situation repeats for inception_v3, googlenet, squeezenet and mobilenet. These models only have a single layer classifier.


class resnet50(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
        super(resnet50, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class resnet101(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet101, pretrained=True):
        super(resnet101, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class resnet152(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet152, pretrained=True):
        super(resnet152, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class resnext50_32x4d(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnext50_32x4d, pretrained=True):
        super(resnext50_32x4d, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class resnext101_32x8d(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnext101_32x8d, pretrained=True):
        super(resnext101_32x8d, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class densenet121(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.densenet121, pretrained=True):
        super(densenet121, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = nn.Linear(
            in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class densenet161(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.densenet161, pretrained=True):
        super(densenet161, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = nn.Linear(
            in_features=2208, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class densenet201(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.densenet201, pretrained=True):
        super(densenet201, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier = nn.Linear(
            in_features=1920, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class efficientnet_b2(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.efficientnet_b2, pretrained=True):
        super(efficientnet_b2, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Linear(
            in_features=1408, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class efficientnet_b4(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.efficientnet_b4, pretrained=True):
        super(efficientnet_b4, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class efficientnet_b5(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.efficientnet_b5, pretrained=True):
        super(efficientnet_b5, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class efficientnet_b7(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.efficientnet_b7, pretrained=True):
        super(efficientnet_b7, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.classifier[1] = nn.Linear(
            in_features=2560, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class regnet_x_8gf(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.regnet_x_8gf, pretrained=True):
        super(regnet_x_8gf, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=1920, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x
