"""Wrappers to sklearn/custom metrics used by the codebase
"""

import numpy as np
from sklearn.metrics import *
from keras import backend as K
import surface_distance


def precision_at_90recall(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores[:, 1])
    idx = np.argmin(np.abs(r - 0.9))
    return p[idx]


def recall_at_90precision(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores[:, 1])
    idx = np.argmin(np.abs(p - 0.9))
    return r[idx]


def dsc(mask_gt, mask_pred, smooth=1e-8):
    """Dice Similarity Coefficient

    Args:
        mask_gt ([type]): [description]
        mask_pred ([type]): [description]
        smooth ([type], optional): [description]. Defaults to 1e-8.

    Returns:
        [type]: [description]
    """
    intersection = K.sum(mask_gt * mask_pred, axis=[1, 2, 3])
    union = K.sum(mask_gt, axis=[1, 2, 3]) + K.sum(mask_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def iou(mask_gt, mask_pred, smooth=1e-8):
    """Intersection over Union

    Args:
        mask_gt ([type]): [description]
        mask_pred ([type]): [description]
        smooth (int, optional): [description]. Defaults to 1e-8.

    Returns:
        [type]: [description]
    """
    intersection = K.sum(K.abs(mask_gt * mask_pred), axis=[1,2,3])
    union = K.sum(mask_gt,[1,2,3])+K.sum(mask_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def surface_distances(mask_gt, mask_pred):
    surface_distances = surface_distance.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(3, 2, 1))
    a = surface_distance.compute_average_surface_distance(
        surface_distances)
    b = surface_distance.compute_robust_hausdorff(
        surface_distances, 100)
    c = surface_distance.compute_surface_overlap_at_tolerance(
        surface_distances, 1)
    d = surface_distance.compute_surface_dice_at_tolerance(
        surface_distances, 1)
    e = surface_distance.compute_dice_coefficient(mask_gt, mask_pred)

    return a, b, c, d, e
