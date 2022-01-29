"""Wrappers to sklearn/custom metrics used by the codebase
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             cohen_kappa_score, confusion_matrix, det_curve,
                             f1_score, fbeta_score, hamming_loss, hinge_loss,
                             jaccard_score, log_loss, matthews_corrcoef,
                             plot_det_curve, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

import segmentation_models_pytorch as smp
smp.metrics.get_stats


def precision_at_90recall(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores[:, 1])
    idx = np.argmin(np.abs(r - 0.9))
    return p[idx]


def recall_at_90precision(gt_labels, pred_scores):
    p, r, _ = precision_recall_curve(gt_labels, pred_scores[:, 1])
    idx = np.argmin(np.abs(p - 0.9))
    return r[idx]