import os
import random
import time

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
        task="semantic-segmentation",
        feature=None,
        sample=None,
        fraction=1.0,
        split_version="v3",
        NUM_CLASSES=2,
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
        self.splitfile_path = os.path.join(
            self.ROOT, "splits", split_version, f"{phase}.txt"
        )
        self.image_list = self.get_image_list(self.splitfile_path)

        # Filter to a fraction of the data for dry-runs
        self.image_list = self.get_dataset_subset(fraction)

        # Number of classes for classification
        self.num_classes = NUM_CLASSES

        # Create labels
        print(f"Creating labels for task {task}")
        if feature is not None:
            assert (
                feature in self.feature_code_to_name_mapping
            ), "FS2: Unknown feature index for code-to-name mapping"
            print(f"Feature : {self.feature_code_to_name_mapping[feature]}")
        self.image2label_dict = self.create_labels(
            self.image_list, task, feature)

        # Get positive and negative indices (for binary class task)
        classwise_indices = self.split_by_class_labels()
        self.neg_indices, self.pos_indices = classwise_indices[0], classwise_indices[1]

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

    def create_labels(self, image_list, task, feature):
        # ROUND2-TODO: Complex function, deserving splitting into parts. Could also be part of the abstraction
        start_time = time.time()
        target_dict = {}
        for _, imagepath in enumerate(tqdm.tqdm(image_list, desc="Creating labels")):
            if task == "disease-classification":
                # TODO: Assumes specific folder structure. May break if that's changed
                # ROUND2-TODO: put "template" paths in utils.constants.
                if imagepath.split("/")[5] == "positive":
                    target_dict[imagepath] = 1
                else:
                    target_dict[imagepath] = 0
            elif task == "feature-classification":
                annot_path = imagepath.replace(".jpg", ".txt")
                with open(annot_path, "r") as f:
                    annotations = f.read().split("\n")
                if annotations == [""]:
                    target_dict[imagepath] = 0
                else:
                    for j in range(len(annotations) - 1):
                        feature_code = int(annotations[j][0])
                        if feature == feature_code:
                            target_dict[imagepath] = 1
                        else:
                            target_dict[imagepath] = 0
            elif task == "feature-detection":
                target_dict[imagepath] = {}
                annot_path = imagepath.replace(".jpg", ".txt")
                with open(annot_path, "r") as f:
                    annotations = f.read().split("\n")
                if annotations == [""]:
                    target_dict[imagepath] = 0
                else:
                    import pdb

                    pdb.set_trace()
            else:
                raise ValueError(f"task {task} is unrecognised.")

        print(
            "Total time to create the labels for task {} : {:.4f}".format(
                task, time.time() - start_time
            )
        )
        return target_dict

    def get_image_list(self, splitfile_path):
        """Read images from splits file and shuffle
        """
        with open(splitfile_path, "r") as f:
            image_list = f.read().split("\n")

        random.shuffle(image_list)
        print("> Found total images : {}".format(len(image_list)))

        return image_list

    def split_by_class_labels(self):
        classwise_indices = [[] for i in range(self.num_classes)]

        for i, image in enumerate(self.image_list):
            class_label = self.image2label_dict[image]
            classwise_indices[class_label].append(i)

        for class_label in range(self.num_classes):
            print(
                "> No. of samples for class {}: {}".format(
                    class_label, len(classwise_indices[class_label])
                )
            )

        return classwise_indices

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
        if fraction != 1.0:
            num_samples = int(fraction * len(self.image_list))
            # ROUND2-TODO: consider a random.shuffle(mylist) followed by sub selection mylist[:self.pos_indices].
            image_list = random.sample(self.image_list, num_samples)

            print("> No. total samples using: {}".format(len(image_list)))
            return image_list
        else:
            print("> No. total samples using: {}".format(len(self.image_list)))
            return self.image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.load_and_process_image(self.image_list[idx], self.transform)
        label = self.image2label_dict[self.image_list[idx]]
        image_path = self.image_list[idx]

        return (image, label, image_path)
