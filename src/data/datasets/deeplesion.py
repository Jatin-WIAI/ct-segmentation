import random
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw

from os.path import join, basename

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


    def create_mask_from_coords(self, recist_coords: np.array, img_shape: tuple):
        points2d = np.array(recist_coords).reshape((-1, 2)).T
        # Create more points using spleen
        tck, u = splprep(points2d)
        unew = np.linspace(0, 1, 100)
        coords = splev(unew, tck)
        coords = np.vstack((coords[0], coords[1])).T

        # Create mask using a dummy PIL Image
        dummy_img = Image.new('L', img_shape, 0)
        ImageDraw.Draw(dummy_img).polygon(
            list(map(tuple, coords)), outline=1, fill=1)
        mask = np.array(dummy_img)

        return mask

    def create_mask(self, imagepath):
        image_fname_csv = '_'.join(imagepath.split('/')[-2:])
        df_qres = self.df_metadata[self.df_metadata['File_name'] == image_fname_csv]
        if len(df_qres) == 0:
            image = plt.imread(imagepath)
            mask = np.zeros(image.shape)
        else:
            
            recist_coords = df_qres.iloc[0]['Measurement_coordinates']
            imageshape = tuple(map(int, df_qres.iloc[0]['Image_size']))

            mask = self.create_mask_from_coords(recist_coords=recist_coords, 
                                                img_shape=imageshape)
            
        return mask

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
        imagepath = self.image_paths[idx]
        image = Image.open(imagepath)
        mask = self.create_mask(imagepath)

        transformed = self.transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        return image, mask
