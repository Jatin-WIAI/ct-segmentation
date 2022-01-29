import numpy as np
import torch

def collate_fn_semantic_seg(batch):
    new_batch = np.empty((len(batch), len(batch[0])), dtype='object')
    new_batch[:, :] = batch 

    images = torch.stack(new_batch[:, 0].tolist())
    masks = torch.stack(new_batch[:, 1].tolist())
    
    return images, masks


def collate_fn_obj_detection(batch):
    tuple(zip(*batch))
