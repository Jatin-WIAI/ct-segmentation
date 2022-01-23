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
from scipy.sparse import csc_matrix
from torch.utils.data import Dataset


class LIDC_IDRI(Dataset):
    """
    Pytorch Dataset object for abdomenCT1k.
    Constructor Args:
        image_mask_mapping (list of str): List of image paths.
        image2label_dict (dict): Dictionary mapping image path to label.
        transform (Pytorch transform object): Transformation to apply on each image.
        phase (str): train/val/test split for data.
    Returns:
        Transformed image, label and image path through __getitem__() function.
    """

    def __init__(self, ROOT='/scratche/users/sansiddh/LIDC-IDRI/data/',
                 transform=None, phase="train", sample=None, fraction=1.0,
                 split_version="v3", task='semantic-segmentation', **kwargs):

        self.ROOT = ROOT
        self.transform = transform
        # Load split file
        if split_version is not None:
            self.splitfile_path = os.path.join(
                self.ROOT, "splits", split_version, f"{phase}.txt"
            )
        else:
            self.splitfile_path = None

        # Get list of Scan objects based on splitfile
        self.all_scans = self.get_list_of_scans(self.splitfile_path)

        # Creates mapping between image filename and image SOP ID
        self.sop_id_to_fname_dict = self.create_imagename_sop_id_mapping()

        # Get dataframe containing all masks and all imagepaths (main func)
        self.image_mask_mapping = self.create_image_masks_mapping()

        # Perform class wise upsampling/downsampling based on config
        # if phase == "train":
        #     if sample is not None:
        #         sampling_object = getattr(self, sample)
        #         self.image_mask_mapping = sampling_object(self)
        
        # TODO : Get fractioning working for pandas dataframes 

    
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

    def create_mask_from_contour(self, contour, img_shape):
        coords = contour.to_matrix(include_k=False)
        coords = np.flip(coords, axis=1)
        dummy_img = Image.new('L', img_shape, 0)
        ImageDraw.Draw(dummy_img).polygon(list(map(tuple, coords)), outline=1, fill=1)
        mask = np.array(dummy_img)

        return mask

    def get_masks_for_single_sample(self, scan):
        images = scan.load_all_dicom_images()
        df_images_masks = pd.DataFrame(columns=['image', 'mask'])
        for i, image in enumerate(images):
            fname = self.sop_id_to_fname_dict[image.SOPInstanceUID]
            df_images_masks.loc[i, 'image'] = fname
            df_images_masks.loc[i, 'mask'] = csc_matrix(
                np.zeros(image.pixel_array.shape))

        all_annotations = scan.annotations
        for annotation in all_annotations:
            contours = annotation.contours
            for contour in contours:
                image = images[contour.image_k_position]
                fname = self.sop_id_to_fname_dict[image.SOPInstanceUID]
                mask = self.create_mask_from_contour(
                    contour, image.pixel_array.shape)
                sparse_mask = csc_matrix(mask)

                idx = df_images_masks[df_images_masks['image'] == fname].index[0]
                org_mask = df_images_masks.loc[idx, 'mask']
                if len(org_mask.nonzero()[0]) == 0:
                    df_images_masks.loc[idx, 'mask'] = sparse_mask
                else:
                    new_idx = len(df_images_masks)
                    df_images_masks.loc[new_idx, 'image'] = fname
                    df_images_masks.loc[new_idx, 'mask'] = sparse_mask

        return df_images_masks

    def create_image_masks_mapping(self):
        df_master = self.get_masks_for_single_sample(self.all_scans[0])
        for scan in self.all_scans[1:]:
            df = self.get_masks_for_single_sample(scan)
            df_master = pd.concat([df_master, df], ignore_index=True)

        return df_master


    def create_imagename_sop_id_mapping(self):
        try:
            df = pd.read_csv('/scratche/users/sansiddh/LIDC-IDRI/fname_sop_id_mapping.csv')
        except Exception:
            print('This will take a long while...')
            all_dicoms = glob(self.ROOT+'LIDC-IDRI-*/*/*/*.dcm')
            df = pd.DataFrame(columns=['filename', 'sop_instance_id'])
            for i, path in tqdm(enumerate(all_dicoms)):
                df.loc[i, 'filename'] = path
                dicom = dcmread(path)
                df.loc[i, 'sop_instance_id'] = dicom.SOPInstanceUID
            
            df.to_csv('/scratche/users/sansiddh/LIDC-IDRI/fname_sop_id_mapping.csv')
        
        sop_id_to_fname_dict = dict(zip(df['sop_instance_id'], df['filename']))

        return sop_id_to_fname_dict


    def get_list_of_scans(self, splitfile_path):
        """Read images from splits file and shuffle
        """
        if splitfile_path is not None:
            with open(splitfile_path, "r") as f:
                patient_list = f.read().split("\n")
            all_scans = pl.query(pl.Scan).filter(
                pl.Scan.patient_id.in_(patient_list)).all()
        else:
            all_scans = pl.query(pl.Scan).all()

        random.shuffle(all_scans)
        print("> Total number of patients : {}".format(len(all_scans)))

        return all_scans

    # def downsampling_negatives(self):
    #     """Assumes negatives are more than positives, and downsamples thme
    #     """

    #     print("Downsampling : negative set = positive set ...")
    #     # ROUND2-TODO: consider a random.shuffle(mylist) followed by sub selection mylist[:self.pos_indices].
    #     downsampled_neg_indices = random.sample(
    #         self.neg_indices, len(self.pos_indices))
    #     downsampled_image_list = list(
    #         np.array(self.image_list)[
    #             self.pos_indices + downsampled_neg_indices]
    #     )

    #     return downsampled_image_list

    # def upsampling_positives(self):
    #     """Assumes positives are fewer than negatives, and upsamples thme
    #     """

    #     print("Upsampling : positive set = negative set ...")
    #     upsample_len = len(self.neg_indices) - len(self.pos_indices)
    #     upsampled_indices = random.choices(self.pos_indices, k=upsample_len)
    #     upsampled_image_list = list(
    #         np.array(self.image_list)[
    #             self.pos_indices + self.neg_indices + upsampled_indices
    #         ]
    #     )

    #     return upsampled_image_list

    def get_dataset_subset(self, fraction):
        num_samples = int(fraction * len(self.image_mask_mapping))
        image_mask_mapping = random.sample(self.image_mask_mapping, num_samples)

        print("> No. total samples using: {}".format(len(image_mask_mapping)))
        return image_mask_mapping

    def __len__(self):
        return len(self.image_mask_mapping)

    def __getitem__(self, idx):
        image_path = self.image_mask_mapping[idx, 'image']
        dicom = dcmread(image_path)
        image = self.convert_dicom_to_image(dicom, return_type='PIL')
        mask = self.image_mask_mapping[idx, 'mask'].toarray()

        transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        return image, mask
