import os
import numpy as np
import nibabel as nib

from glob import glob

from PIL import Image
from os.path import basename


ROOT_DIR = '/scratche/users/sansiddh/abdomenCT-1k/'
WINDOWING_LEN = 300
all_masks = glob(ROOT_DIR+'Masks/*.nii.gz')
len(all_masks)

for mask in all_masks:
    ctscan_path = mask.replace('Masks', 'Cases')
    ctscan_path = ctscan_path.replace('.nii.gz', '_0000.nii.gz')

    ctscan = nib.load(ctscan_path).get_fdata()
    for i in range(ctscan.shape[2]):
        ctscan_img = ctscan[:, :, i]
        ctscan_img[ctscan_img > WINDOWING_LEN] = WINDOWING_LEN
        ctscan_img[ctscan_img < -WINDOWING_LEN] = -WINDOWING_LEN
        hounsfield_mask = np.rot90(ctscan_img, k=1)

        gsimg_channel = ((hounsfield_mask + WINDOWING_LEN) *
                         255/(2*WINDOWING_LEN)).astype(np.uint8)
        img = np.repeat(gsimg_channel[:, :, np.newaxis], 3, axis=2)

        img = Image.fromarray(img)
        id = basename(mask).split('_')[1].split('.')[0]
        imagesdir = f'{ROOT_DIR}Images/{id}/'
        os.makedirs(imagesdir, exist_ok=True)

        img.save(imagesdir + '%05d.jpg' % i)

    print(f'{mask} done')
