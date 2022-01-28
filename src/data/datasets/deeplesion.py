import copy
import pickle
import random
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
from pydicom import dcmread
from os.path import join, basename
from scipy.sparse import csc_matrix
from scipy.interpolate import splprep, splev
from torch.utils.data import Dataset


class DeepLesion(Dataset):
    """
    Pytorch Dataset object for LIDC_IDRI
    Constructor Args:
        image_mask_mapping (list of str): List of image paths.
        image2label_dict (dict): Dictionary mapping image path to label.
        transform (Pytorch transform object): Transformation to apply on each image.
        phase (str): train/val/test split for data.
    Returns:
        Transformed image, label and image path through __getitem__() function.
    """

    def __init__(self, ROOT='/scratche/users/sansiddh/DeepLesion/',
                 transform=None, phase="train", sample=None, fraction=1.0,
                 split_version="v3", task='semantic-segmentation', **kwargs):

        self.ROOT = ROOT
        self.transform = transform
        self.img_dir = join(ROOT, 'Images_png')
        self.metadata_fname = join(ROOT, 'DL_info.csv')
        self.df_metadata = pd.read_csv(self.metadata_fname)
        # Preprocess CSV
        columns = ['Measurement_coordinates', 'Bounding_boxes', 'Lesion_diameters_Pixel_', 
                   'Normalized_lesion_location', 'Slice_range', 'Spacing_mm_px_', 'Image_size', 
                   'DICOM_windows']
        for colname in columns:
            self.df_metadata[colname] = self.df_metadata[colname].apply(
                lambda x: list(map(float, x.split(', '))))

        # Load split file
        if split_version is not None:
            self.splitfile_path = join(
                self.ROOT, "splits", split_version, f"{phase}.txt"
            )
        else:
            self.splitfile_path = None

        # Get list of Scan objects based on splitfile
        self.image_paths = self.get_list_of_images(self.splitfile_path)

        # Get dataframe containing all masks and all imagepaths (main func)
        self.image_mask_mapping = self.create_masks(self.image_paths)

        # Perform class wise upsampling/downsampling based on config
        # if phase == "train":
        #     if sample is not None:
        #         sampling_object = getattr(self, sample)
        #         self.image_mask_mapping = sampling_object(self)

        # TODO : Get fractioning working for pandas dataframes

    # def convert_dicom_to_image(self, dicom, return_type='numpy'):
    #     window_center = int(dicom.WindowCenter)
    #     window_width = int(dicom.WindowWidth)
    #     hu_values = dicom.pixel_array

    #     min_hu, max_hu = (window_center - window_width/2,
    #                       window_center + window_width/2)

    #     img = copy.copy(hu_values)
    #     img[img > max_hu] = max_hu
    #     img[img < min_hu] = min_hu
    #     img = (img - min_hu)/(max_hu - min_hu)

    #     img = (img*255).astype(np.int32)
    #     img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    #     if return_type == 'PIL':
    #         img = img.astype(np.uint8)
    #         return Image.fromarray(img)
    #     elif return_type == 'numpy':
    #         return img
    #     else:
    #         raise ValueError('`return_type` can be one of PIL or numpy')

    def create_mask_from_coords(self, coords: np.array, img_shape: tuple):
        dummy_img = Image.new('L', img_shape, 0)
        ImageDraw.Draw(dummy_img).polygon(
            list(map(tuple, coords)), outline=1, fill=1)
        mask = np.array(dummy_img)

        return mask

    # def get_masks_for_single_sample(self, scan):
    #     images = scan.load_all_dicom_images()
    #     df_masks = pd.DataFrame(columns=['image', 'mask'])
    #     df_attrs = pd.DataFrame(columns=['image'])
    #     attributes = ['subtlety', 'internalStructure', 'calcification', 'sphericity',
    #                   'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
    #     for i, image in enumerate(images):
    #         fname = self.sop_id_to_fname_dict[image.SOPInstanceUID]
    #         df_masks.loc[i, 'image'] = fname
    #         # Create empty mask for all images first
    #         df_masks.loc[i, 'mask'] = csc_matrix(
    #             np.zeros(image.pixel_array.shape))

    #     # Create dummy attribute entries for all attributes
    #     for i, image in enumerate(images):
    #         fname = self.sop_id_to_fname_dict[image.SOPInstanceUID]
    #         df_attrs.loc[i, 'image'] = fname
    #         for attr in attributes:
    #             df_attrs.loc[i, attr] = -1
    #             title_attr = 'InternalStructure' if attr == 'internalStructure' else attr.title()
    #             df_attrs.loc[i, title_attr] = ""

    #     all_annotations = scan.annotations
    #     for annotation in all_annotations:
    #         contours = annotation.contours
    #         for contour in contours:
    #             image = images[contour.image_k_position]
    #             fname = self.sop_id_to_fname_dict[image.SOPInstanceUID]
    #             mask = self.create_mask_from_contour(
    #                 contour, image.pixel_array.shape)
    #             sparse_mask = csc_matrix(mask)

    #             idx = df_masks[df_masks['image'] == fname].index[0]
    #             org_mask = df_masks.loc[idx, 'mask']
    #             # Replace empty mask with generated mask if empty masks exists
    #             # Else create new row entry with given mask
    #             if len(org_mask.nonzero()[0]) == 0:
    #                 df_masks.loc[idx, 'mask'] = sparse_mask
    #                 for attr in attributes:
    #                     df_attrs.loc[idx, attr] = getattr(
    #                         annotation, attr)
    #                     title_attr = 'InternalStructure' if attr == 'internalStructure' else attr.title()
    #                     if (attr == 'internalStructure') & (df_attrs.loc[idx, attr] not in range(1, 5)):
    #                         pass
    #                     else:
    #                         df_attrs.loc[idx, title_attr] = getattr(
    #                             annotation, title_attr)
    #             else:
    #                 new_idx = len(df_masks)
    #                 df_masks.loc[new_idx, 'image'] = fname
    #                 df_attrs.loc[new_idx, 'image'] = fname
    #                 df_masks.loc[new_idx, 'mask'] = sparse_mask
    #                 for attr in attributes:
    #                     df_attrs.loc[new_idx, attr] = getattr(
    #                         annotation, attr)
    #                     title_attr = 'InternalStructure' if attr == 'internalStructure' else attr.title()
    #                     if (attr == 'internalStructure') & (df_attrs.loc[new_idx, attr] not in range(1, 5)):
    #                         pass
    #                     else:
    #                         df_attrs.loc[new_idx, title_attr] = getattr(
    #                             annotation, title_attr)

    #     return df_masks, df_attrs

    def create_masks(self, image_paths):
        image_mask_mapping = {}
        for imagepath in tqdm(image_paths, desc="Creating masks for N Images"):
            image_fname_csv = '_'.join(imagepath.split('/')[-2:])
            df_qres = self.df_metadata[self.df_metadata['File_name'] == image_fname_csv]
            if len(df_qres) == 0:
                image = plt.imread(imagepath)
                sparse_mask = csc_matrix(np.zeros(image.shape))
            else:
                points2d = np.array(df_qres.iloc[0]['Measurement_coordinates']).reshape((-1, 2)).T
                tck, u = splprep(points2d)
                unew = np.linspace(0, 1, 100)
                coords = splev(unew, tck)
                coords = np.vstack((coords[0], coords[1])).T

                imageshape = tuple(map(int, df_qres.iloc[0]['Image_size']))

                mask = self.create_mask_from_coords(coords=coords, img_shape=imageshape)
                sparse_mask = csc_matrix(mask)

            image_mask_mapping[imagepath] = sparse_mask

    # def create_imagename_sop_id_mapping(self):
    #     try:
    #         df = pd.read_csv(
    #             '/scratche/users/sansiddh/LIDC-IDRI/fname_sop_id_mapping.csv')
    #     except Exception:
    #         print('This will take a long while...')
    #         all_dicoms = glob(self.ROOT+'LIDC-IDRI-*/*/*/*.dcm')
    #         df = pd.DataFrame(columns=['filename', 'sop_instance_id'])
    #         for i, path in tqdm(enumerate(all_dicoms)):
    #             df.loc[i, 'filename'] = path
    #             dicom = dcmread(path)
    #             df.loc[i, 'sop_instance_id'] = dicom.SOPInstanceUID

    #         df.to_csv(
    #             '/scratche/users/sansiddh/LIDC-IDRI/fname_sop_id_mapping.csv')

    #     sop_id_to_fname_dict = dict(zip(df['sop_instance_id'], df['filename']))

    #     return sop_id_to_fname_dict

    def get_list_of_images(self, splitfile_path):
        """Read images from splits file and shuffle
        """
        if splitfile_path is not None:
            with open(splitfile_path, "r") as f:
                all_patients = f.read().split("\n")
            all_patients = [int(x) for x in all_patients]
            df_subset = self.df_metadata[self.df_metadata['Patient_index'].isin(all_patients)]
            folder_names = [f'{row["Patient_index"]}_{row["Study_index"]}_{row["Series_ID"]}' for row in df_subset.itterows()]
            image_paths = []
            for folder in tqdm(folder_names, desc="Num Patients globbing imgs for"):
                image_paths.append(glob(join(self.img_dir, folder, '*.png')))

        else:
            print("Globbing, will take a while.")
            all_patients_rep = glob(join(self.img_dir, '*'))
            all_patients_rep = [basename(x) for x in all_patients_rep]
            all_patients = np.unique(all_patients_rep)
            image_paths = glob(join(self.img_dir, '*/*.png'))
            df_subset = self.df_metadata

        random.shuffle(image_paths)
        print("> Total number of patients : {}".format(len(all_patients)))
        print("> Total number of images : {}".format(len(image_paths)))
        print("> Total number of annotated images : {}".format(len(df_subset)))

        return image_paths

    def get_dataset_subset(self, fraction):
        num_samples = int(fraction * len(self.image_mask_mapping))
        image_mask_mapping = random.sample(
            self.image_mask_mapping, num_samples)

        print("> No. total samples using: {}".format(len(image_mask_mapping)))
        return image_mask_mapping

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_mask_mapping[idx, 'image']
        dicom = dcmread(image_path)
        image = self.convert_dicom_to_image(dicom, return_type='PIL')
        mask = self.image_mask_mapping[idx, 'mask'].toarray()

        transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        return image, mask
