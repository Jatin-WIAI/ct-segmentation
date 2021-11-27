import os
import random
import time
from glob import glob

import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset


class abdomenCT1k(Dataset):
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

    def __init__(
        self,
        ROOT='/scratche/users/sansiddh/abdomenCT-1k/',
        transform=None,
        phase="train",
        sample=None,
        fraction=1.0,
        split_version="v3",
        num_classes=2,
        **kwargs,
    ):

        self.ROOT = ROOT
        self.transform = transform

        self.feature_code_to_name_mapping = {
            1 : 'liver',
            2 : 'kidney',
            3 : 'spleen',
            4 : 'pancreas'
        }

        # Loads images list
        if split_version is not None:
            self.splitfile_path = os.path.join(
                self.ROOT, "splits", split_version, f"{phase}.txt"
            )
        else:
            self.splitfile_path = None
        
        # Gets images from splitfile path. If splitfile path is None, loads all images
        self.image_list = self.get_image_list(self.splitfile_path)

        # Filter to a fraction of the data for dry-runs
        if fraction != 1:
            self.image_list = self.get_dataset_subset(fraction)
        print("> No. total samples using: {}".format(len(self.image_list)))

        # Number of classes for classification
        self.num_classes = num_classes

        if phase == "train":
            # TODO: Should this disabled by default?
            # TODO: Check this implementation for bugs.
            if sample is not None:
                sampling_object = getattr(abdomenCT1k, sample)
                self.image_list = sampling_object(self)

    def load_and_process_image(self, image_name, transform):
        """
        Load and apply transformation to image.
        """
        image = Image.open(image_name)
        img = transform(image)
        image.close()
        return img

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
        mask_path = image_path.replace('.jpg', '.npy').replace('Images', 'Masks')
        image = self.load_and_process_image(image_path, self.transform)
        mask = np.load(mask_path)

        target = {
            'image': image,
            'mask': mask
        }

        return target
