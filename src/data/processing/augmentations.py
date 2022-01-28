import sys

import albumentations as A
from albumentations import (
    Affine,
    Blur,
    CenterCrop,
    Crop,
    CropAndPad,
    ElasticTransform,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomCrop,
    RandomResizedCrop,
    RandomRotate90,
    RandomScale,
    RandomSizedCrop,
    Resize,
    Rotate,
)
from albumentations.pytorch import ToTensorV2

"""
Check out https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/
for documentation on all cropping augmentations

Geometric Resize transforms - https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/

Geometric Transforms - https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/

All other transforms - https://albumentations.ai/docs/api_reference/augmentations/transforms/
"""

"""
Example docstrings - 

class albumentations.augmentations.crops.transforms.RandomCrop (height, width, always_apply=False, p=1.0)
height	int	
height of the crop.

width	int	
width of the crop.

p	float	
probability of applying the transform. Default: 1.
"""


class Augmentations():
    def __init__(self, cfg, use_augmentation=False, size=[224, 224]):
        self.size = tuple(size)
        augs = []
        augs.append(Resize(*self.size))
        if use_augmentation:
            augs.extend(self.get_augmentations(cfg['augs']))

        augs.append(Normalize(**cfg['normalization']))
        augs.append(ToTensorV2())

        if cfg['bbox_aug']:
            self.augs = A.Compose(
                augs, bbox_params=A.BboxParams(**cfg['bbox_aug_params']))
        else:
            self.augs = A.Compose(augs)

    def __call__(self, image, bboxes=None, class_labels=None, mask=None):
        if bboxes is None and mask is None:
            return self.augs(image=image)
        elif bboxes is not None and mask is None:
            return self.augs(image=image, bboxes=bboxes, class_labels=class_labels)
        elif bboxes is None and mask is not None:
            return self.augs(image=image, mask=mask)
        else:
            return self.augs(image=image, bboxes=bboxes, class_labels=class_labels, mask=mask)

    def get_augmentations(self, cfg):
        augs = []
        for augmentation in cfg:
            augmentation_object = getattr(sys.modules[__name__], augmentation)
            augmentation_param_dict = cfg[augmentation]
            augs.append(augmentation_object(**augmentation_param_dict))

        return augs
