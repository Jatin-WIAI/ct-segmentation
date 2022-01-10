# Augmentations using `imgaug` python library
# Reference - https://imgaug.readthedocs.io/en/latest/
# Actual effect of augmentations can be seen through the notebook augmentation_testing.ipynb

import sys

import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import albumentations as A
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image

ia.seed(0)


class AvgBlur:
    def __init__(self, k):
        self.aug = iaa.AverageBlur(k=k)

    def __call__(self, img):
        return self.aug.augment_image(img)


class MedianBlur:
    def __init__(self, k):
        self.aug = iaa.MedianBlur(k=k)

    def __call__(self, img):
        return self.aug.augment_image(img)


class GaussianBlur:
    def __init__(self, sigma):
        self.aug = iaa.GaussianBlur(sigma=sigma)

    def __call__(self, img):
        return self.aug.augment_image(img)


class BilateralBlur:
    def __init__(self, d):
        self.aug = iaa.BilateralBlur(d=d)

    def __call__(self, img):
        return self.aug.augment_image(img)


class NpConvert:
    def __init__(self, ):
        pass

    def __call__(self, img):
        print(type(img))
        img = np.array(img)
        return img


class ImConvert:
    def __init__(self, ):
        pass

    def __call__(self, img):
        img = Image.fromarray(img)
        return img


class MotionBlur:
    def __init__(self, k):
        self.aug = iaa.MotionBlur(k=k)

    def __call__(self, img):
        return self.aug.augment_image(img)


class Brightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        return F.adjust_brightness(x, self.brightness_factor)


class Contrast:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, x):
        return F.adjust_contrast(x, self.contrast_factor)


class AddElementWise:
    def __init__(self, value):
        self.aug = iaa.AddElementwise(value=value)

    def __call__(self, img):
        return self.aug.augment_image(img)


class GaussianNoise:
    def __init__(self, scale):
        self.aug = iaa.AdditiveGaussianNoise(scale=scale)

    def __call__(self, img):
        return self.aug.augment_image(img)


class SaltAndPepper:
    def __init__(self, p):
        self.aug = iaa.SaltAndPepper(p=p)

    def __call__(self, img):
        return self.aug.augment_image(img)


class Sharpen:
    def __init__(self, alpha, lightness):
        self.aug = iaa.Sharpen(alpha=alpha, lightness=lightness)

    def __call__(self, img):
        return self.aug.augment_image(img)


class RandomHorizontalFlip():
    def __init__(self, flip_prob=0.25):
        self.seq = iaa.Fliplr(flip_prob)

    def __call__(self, image):
        return self.seq.augment_image(image)


class PhotometricDistort():
    def __init__(self, prob=0.25, hue_add_range=(-50, 50), saturation_add_range=(-75, 75),
                 gamma_range=(0.5, 1.75), log_range=(0.5, 1.5), linear_range=(0.5, 1.75)):
        self.seq = iaa.Sometimes(prob, iaa.Sequential([iaa.OneOf([
            iaa.SomeOf((1, 2), [
                iaa.MultiplyHue(from_colorspace="BGR"),
                iaa.MultiplySaturation(from_colorspace="BGR"),
            ], random_order=True),
            iaa.SomeOf((1, 2), [
                iaa.AddToHue(value=hue_add_range,
                             from_colorspace="BGR"),
                iaa.AddToSaturation(
                    value=saturation_add_range, from_colorspace="BGR"),
            ], random_order=True),
            iaa.OneOf([
                iaa.GammaContrast(gamma_range),
                iaa.LogContrast(log_range),
                iaa.LinearContrast(linear_range),
                iaa.AllChannelsHistogramEqualization(),
            ])
        ])
        ]))

    def __call__(self, image):
        return self.seq.augment_image(image)


class RandomAddMultiply():
    def __init__(self, prob=0.2, add_limit=25, multiply_range=(0.75, 1.25)):
        add = iaa.Add((-add_limit, add_limit))
        multiply = iaa.Multiply(multiply_range)

        self.seq = iaa.Sometimes(
            prob, iaa.Sequential([iaa.OneOf([add, multiply])]))

    def __call__(self, image):
        return self.seq.augment_image(image)


class RandomBlur():
    def __init__(self, prob=0.1, sigma_range=(0, 3.0), k=5):
        self.prob = prob
        self.sigma_range = sigma_range
        self.k = k

    def __call__(self, image):
        seq = iaa.Sometimes(self.prob, iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.GaussianBlur(self.sigma_range),
                    iaa.MotionBlur(k=10),
                    iaa.MedianBlur(self.k),
                    iaa.AverageBlur(self.k),
                    iaa.BilateralBlur(d=2),
                ])
            ]))
        image = seq.augment_image(image)
        return image


class RandomNoise(object):
    def __init__(self, prob=0.1, scale=0.1, p=0.05):
        self.prob = prob
        self.scale = scale * 255
        self.p = p

    def __call__(self, image):
        seq = iaa.Sometimes(self.prob, iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=self.scale),
                    iaa.SaltAndPepper(p=self.p),
                ])
            ]))
        image = seq.augment_image(image)
        return image


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for _transform in self.transforms:
            image = _transform(image)
        return image


class Augmentations():
    def __init__(self, cfg, use_augmentation=False, size=[224, 224]):
        self.size = tuple(size)
        augs = []
        augs.append(transforms.Resize(self.size))
        if use_augmentation:
            augs.append(NpConvert())
            augs.extend(self.get_augmentations(cfg['main_augs']))
            augs.append(ImConvert())
            augs.extend(self.get_augmentations(cfg['misc_augs']))

        augs.append(transforms.ToTensor())
        augs.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.augs = Compose(augs)

    def __call__(self, image, mask=None):
        if mask is not None:
            return self.augs(image), self.augs(mask)
        else:
            return self.augs(image)

    def get_augmentations(self, cfg):
        augs = []
        for augmentation in cfg:
            augmentation_object = getattr(sys.modules[__name__], augmentation)
            augmentation_param_dict = cfg[augmentation]
            augs.append(augmentation_object(**augmentation_param_dict))

        return augs
