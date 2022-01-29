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