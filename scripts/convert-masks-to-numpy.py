import os
import numpy as np
import nibabel as nib

from glob import glob

from PIL import Image
from os.path import basename


ROOT_DIR = '/scratche/users/sansiddh/abdomenCT-1k/'
all_masks = glob(ROOT_DIR+'Masks/*.nii.gz')
CLASSES = [0, 1, 2, 3, 4]
print(len(all_masks))

for mask_path in all_masks:

    mask = nib.load(mask_path).get_fdata()
    for i in range(mask.shape[2]):
        mask_img = mask[:, :, i]
        mask_np = np.repeat(mask_img[:, :, np.newaxis], len(CLASSES), axis=2)
        for idx, class_val in enumerate(CLASSES):
            bool_arr = (mask_img == class_val)
            mask_np[:, :, idx][bool_arr] = 1
            mask_np[:, :, idx][~bool_arr] = 0
        
        id = basename(mask_path).split('_')[1].split('.')[0]
        masks_dir = f'{ROOT_DIR}Masks/{id}/'
        os.makedirs(masks_dir, exist_ok=True)

        with open(masks_dir + '%05d.npy' % i, 'wb') as f:
            np.save(f, mask_np)

    print(f'{mask_path} done')
