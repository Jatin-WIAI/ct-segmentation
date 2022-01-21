import copy
import os
import random
import time
from glob import glob

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylidc as pl
import tqdm
from PIL import Image, ImageDraw
from pydicom import dcmread
from torch.utils.data import Dataset


class LIDC_IDRI(Dataset):
    """
    Pytorch Dataset object for abdomenCT1k.
    Constructor Args:
        image_list (list of str): List of image paths.
        image2label_dict (dict): Dictionary mapping image path to label.
        transform (Pytorch transform object): Transformation to apply on each image.
        phase (str): train/val/test split for data.
    Returns:
        Transformed image, label and image path through __getitem__() function.
    """

    def __init__(self, transform=None, phase="train", sample=None, fraction=1.0, 
                 split_version="v3", task='semantic-segmentation', **kwargs):

        # Loads images list
        if split_version is not None:
            self.splitfile_path = os.path.join(
                self.ROOT, "splits", split_version, f"{phase}.txt"
            )
        else:
            self.splitfile_path = None

        

        if phase == "train":
            # TODO: Should this disabled by default?
            # TODO: Check this implementation for bugs.
            if sample is not None:
                sampling_object = getattr(self, sample)
                self.image_list = sampling_object(self)

    
    def convert_dicom_to_image(self, dicom, return_type='numpy'):
        window_center = int(dicom.WindowCenter)
        window_width = int(dicom.WindowWidth)
        hu_values = dicom.pixel_array

        min_hu, max_hu = (window_center - window_width/2, 
                          window_center + window_width/2)
        
        img = copy.copy(hu_values)
        img[img > max_hu] = max_hu
        img[img < min_hu] = min_hu
        img = (img - min_hu)/(max_hu - min_hu)
        
        img = (img*255).astype(np.int32)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if return_type == 'PIL':
            img = img.astype(np.uint8)
            return Image.fromarray(img)
        elif return_type == 'numpy':
            return img
        else:
            raise ValueError('`return_type` can be one of PIL or numpy')

    def create_mask_from_contour(self, contour, width=512, height=512):
        coords = contour.to_matrix(include_k=False)
        coords = np.flip(coords, axis=1)
        dummy_img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(dummy_img).polygon(list(map(tuple, coords)), outline=1, fill=1)
        mask = np.array(dummy_img)

        return mask



    def get_image_list(self, splitfile_path):
        """Read images from splits file and shuffle
        """
        if splitfile_path is not None:
            with open(splitfile_path, "r") as f:
                image_list = f.read().split("\n")
        else:
            image_list = glob(f'{self.ROOT}Images/*/*.jpg')

        random.shuffle(image_list)
        print("> Found total images : {}".format(len(image_list)))

        return image_list

    def downsampling_negatives(self):
        """Assumes negatives are more than positives, and downsamples thme
        """

        print("Downsampling : negative set = positive set ...")
        # ROUND2-TODO: consider a random.shuffle(mylist) followed by sub selection mylist[:self.pos_indices].
        downsampled_neg_indices = random.sample(
            self.neg_indices, len(self.pos_indices))
        downsampled_image_list = list(
            np.array(self.image_list)[
                self.pos_indices + downsampled_neg_indices]
        )

        return downsampled_image_list

    def upsampling_positives(self):
        """Assumes positives are fewer than negatives, and upsamples thme
        """

        print("Upsampling : positive set = negative set ...")
        upsample_len = len(self.neg_indices) - len(self.pos_indices)
        upsampled_indices = random.choices(self.pos_indices, k=upsample_len)
        upsampled_image_list = list(
            np.array(self.image_list)[
                self.pos_indices + self.neg_indices + upsampled_indices
            ]
        )

        return upsampled_image_list

    def get_dataset_subset(self, fraction):
        num_samples = int(fraction * len(self.image_list))
        # ROUND2-TODO: consider a random.shuffle(mylist) followed by sub selection mylist[:self.pos_indices].
        # SJ : This is already happening, we do random.shuffle(list) when we load the images
        image_list = random.sample(self.image_list, num_samples)

        print("> No. total samples using: {}".format(len(image_list)))
        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = image_path.replace(
            '.jpg', '.npy').replace('Images', 'Masks')
        # image = self.load_and_process_image(image_path, self.transform)
        image = Image.open(image_path)
        mask = np.load(mask_path)
        transformed = self.transform(image=image, mask=mask)
        # print(self.transform)
        image = transformed['image']
        mask = transformed['mask']

        print(image.shape)
        return image, mask
