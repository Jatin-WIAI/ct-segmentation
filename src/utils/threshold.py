""" This module provides functions to calculate the optimal threshold for the metric curves.
    Each function returns to optimal threshold value and idx.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def get_best_threshold_fowlkes_mallows(y_val, y_pred_val_proba, pos_label="Yes"):
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_pred_val_proba[:, 1], pos_label=pos_label
    )
    fowlkes_mallows_idx = np.sqrt(precision * recall)
    idx = np.argmax(fowlkes_mallows_idx)
    best_thresh = thresholds[idx]

    return best_thresh, idx


def get_best_threshold_f1(y_val, y_pred_val_proba, pos_label="Yes"):
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_pred_val_proba[:, 1], pos_label=pos_label
    )
    f1 = 2 * (precision * recall / (precision + recall))
    f1[np.isnan(f1)] = 0
    idx = np.argmax(f1)
    best_thresh = thresholds[idx]

    return best_thresh, idx


def get_best_threshold_fbeta(y_val, y_pred_val_proba, beta=1, pos_label="Yes"):
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_pred_val_proba[:, 1], pos_label=pos_label
    )
    fbeta = ((1 + beta ** 2) * precision * recall) / \
        ((beta ** 2 * precision) + recall)
    fbeta[np.isnan(fbeta)] = 0
    idx = np.argmax(fbeta)
    best_thresh = thresholds[idx]

    return best_thresh, idx


def get_best_threshold_gmean(y_val, y_pred_val_proba, pos_label="Yes"):
    fpr, tpr, thresholds = roc_curve(
        y_val, y_pred_val_proba[:, 1], pos_label=pos_label)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    best_thresh = thresholds[idx]

    return best_thresh, idx
