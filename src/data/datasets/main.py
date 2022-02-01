import time

import src.data.datasets as datasets_module
from src.data.processing.augmentations import Augmentations
from src.utils.collation import collate_fn_obj_detection, collate_fn_semantic_seg
from torch.utils.data import DataLoader


def create_dataloader(data_cfg, phase):
    """
    Returns dataloader corresponding to the Dataset object for fs2_data.
    """
    start = time.time()
    print('Preparing {} dataloader...'.format(phase))

    phase_flag = True if phase == 'train' else False
    transform = Augmentations(cfg=data_cfg['augmentations'], use_augmentation=phase_flag,
                              size=data_cfg['imgsize'])
    dataset_class = getattr(datasets_module, data_cfg['dataset'])
    dataset = dataset_class(transform=transform,
                            phase=phase, **data_cfg['dataset_params'])
    end = time.time()

    dataloader = DataLoader(dataset=dataset, collate_fn=collate_fn_semantic_seg,
                            **data_cfg['dataloader_params'])

    print('> Time to create dataset object : {:.4f} sec'.format(end-start))
    print('> Time to create dataloader object : {:.4f} sec'.format(
        time.time() - end))

    return dataloader
