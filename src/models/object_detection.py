import torch
import torchvision.models.detection as tvdetection
import torch.nn as nn

#! NOTE - All torchvision models expect the bounding boxes in pascal_voc format, that is
""" [xmin, ymin, xmax, ymax]. The units are in pixels.
"""


class fasterrcnn_resnet50_fpn(nn.Module):
    def __init__(self, basemodel=tvdetection.fasterrcnn_resnet50_fpn, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(fasterrcnn_resnet50_fpn, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class ssd300_vgg16(nn.Module):
    def __init__(self, basemodel=tvdetection.ssd300_vgg16, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(ssd300_vgg16, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class fasterrcnn_mobilenet_v3_large_fpn(nn.Module):
    def __init__(self, basemodel=tvdetection.fasterrcnn_mobilenet_v3_large_fpn, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(fasterrcnn_mobilenet_v3_large_fpn, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class fasterrcnn_mobilenet_v3_large_320_fpn(nn.Module):
    def __init__(self, basemodel=tvdetection.fasterrcnn_mobilenet_v3_large_320_fpn, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(fasterrcnn_mobilenet_v3_large_320_fpn, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class retinanet_resnet50_fpn(nn.Module):
    def __init__(self, basemodel=tvdetection.retinanet_resnet50_fpn, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(retinanet_resnet50_fpn, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class ssdlite320_mobilenet_v3_large(nn.Module):
    def __init__(self, basemodel=tvdetection.ssdlite320_mobilenet_v3_large, pretrained=False,
                 pretrained_backbone=True, num_classes=5, trainable_backbone_layers=5):
        super(ssdlite320_mobilenet_v3_large, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained, progress=True,
                                   num_classes=num_classes,
                                   pretrained_backbone=pretrained_backbone,
                                   trainable_backbone_layers=trainable_backbone_layers)

    def forward(self, x):
        x = self.basemodel(x)
        return x


#! NOTE - YOLO models expect the bounding boxes in the YOLO format. That is,
""" Each row is class x_center y_center width height format.
    Box coordinates must be in normalized xywh format(from 0 - 1). If your boxes are in pixels, 
    divide x_center and width by image width, and y_center and height by image height.
    Class numbers are zero -indexed (start from 0).
"""


class YOLOv5(nn.Module):
    def __init__(self, model_name='yolov5m', num_classes=10, num_channels=3, pretrained=True):
        """Function which initialises a YOLOv5 model.

        Args:
            model_name (str, optional): The specific YOLOv5 model to use. Defaults to 'yolov5m'.
            Model name can be one of : yolov5s, yolov5m, yolov5l, yolov5x, 
            yolov5n, yolov5n6, yolov5s6, yolov5m6, yolov5l6, yolov5x6
            Check out https://github.com/ultralytics/yolov5/releases for more details
            num_classes (int, optional): Number of output classes. Defaults to 10.
            num_channels (int, optional): Number of input channels. Defaults to 3.
            pretrained (bool, optional): Whether the model is pretrained or not. Defaults to True.
        """
        super(YOLOv5, self).__init__()
        self.basemodel = torch.hub.load('ultralytics/yolov5',
                                        model_name,
                                        classes=num_classes,
                                        channels=num_channels,
                                        pretrained=pretrained)

    def forward(self, x):
        x = self.basemodel(x)
        return x


#! NOTE - YOLO models expect the bounding boxes in the YOLO format. That is,
""" Each row is class x_center y_center width height format.
    Box coordinates must be in normalized xywh format(from 0 - 1). If your boxes are in pixels, 
    divide x_center and width by image width, and y_center and height by image height.
    Class numbers are zero -indexed (start from 0).
"""


class YOLOv3(nn.Module):
    def __init__(self, model_name='yolov3', num_classes=10, num_channels=3, pretrained=True):
        """Function which initialises a YOLOv5 model.

        Args:
            model_name (str, optional): The specific YOLOv5 model to use. Defaults to 'yolov5m'.
            Model name can be one of : yolov3, yolov3-spp, yolov3-tiny
            Check out https://github.com/ultralytics/yolov3/releases for more details
            num_classes (int, optional): Number of output classes. Defaults to 10.
            num_channels (int, optional): Number of input channels. Defaults to 3.
            pretrained (bool, optional): Whether the model is pretrained or not. Defaults to True.
        """
        super(YOLOv5, self).__init__()
        self.basemodel = torch.hub.load('ultralytics/yolov3',
                                        model_name=model_name,
                                        classes=num_classes,
                                        channels=num_channels,
                                        pretrained=pretrained)

    def forward(self, x):
        x = self.basemodel(x)
        return x


"""
#TODO later
For implementing EfficientDet in Pytorch, follow - 
https://gist.github.com/Chris-hughes10/73628b1d8d6fc7d359b3dcbbbb8869d7
"""

"""
#TODO later
Add support for YOLOR - https://github.com/WongKinYiu/yolor
Add support for Scaled-YOLOv4 - https://github.com/WongKinYiu/ScaledYOLOv4
Add support for CenterNet2 - https://github.com/xingyizhou/CenterNet2
"""
