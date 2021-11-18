from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import wandb
from src.utils.threshold import *
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, det_curve,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, roc_auc_score, roc_curve)


def plot_reliability_curve(y_true_arr, y_pred_proba_arr, labels_arr, n_bins=10, log_wandb=False, **kwargs):
    """Function for plotting reliability curve (test of extent of calibration)
    The user provides a list of N GT arrays, N predicted probabilities, N labels, 
    and that results in N reliability curves.

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        n_bins (int, optional): Number of bins to use. Defaults to 10.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: Tuple of fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([0, 1], [0, 1], '--', c='black', label='Perfect Calibration')

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        frac_pos, pred_prob = calibration_curve(
            y_true, y_pred_proba[:, 1], n_bins=n_bins)
        ax.plot(pred_prob, frac_pos, '-o', label=f'{labels_arr[i]}')

    ax.set_title('Reliability Diagram')
    ax.set_xlabel('Predicted Probability, Positive Class')
    ax.set_ylabel('Fraction of Positives')
    ax.grid()
    ax.legend()

    if log_wandb:
        wandb.log({"reliab_curve": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_pr_curve(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
                  plot_prevalance_for_idx=None, log_wandb=False):
    """Function for plotting PR curve

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the PR curve corresponding to this index. Defaults to None.
        plot_prevalance_for_idx (int, optional): If true, prevelance of pos class is 
        plotted for the PR curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        p, r, _ = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        ap = average_precision_score(y_true, y_pred_proba[:, 1],
                                     pos_label=pos_label)
        ax.plot(r, p, label=f'{labels_arr[i]} (AP - {round(ap, 2)})')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_f1(
            y_true, y_pred_proba, pos_label=pos_label)
        p, r, _ = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)

        ax.plot([r[idx]], [p[idx]], '-o', c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (F1)')

    if plot_prevalance_for_idx is not None:
        y_true = y_true_arr[plot_prevalance_for_idx]
        ax.plot([0, 1], [sum(y_true == pos_label)/len(y_true)]*2, c='black', ls='--',
                label=f'{labels_arr[plot_prevalance_for_idx]} Prevelance ' +
                f'({round(sum(y_true == pos_label)/len(y_true), 2)})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve')
    ax.grid()
    ax.legend()
    if log_wandb:
        wandb.log({"pr_curve": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_roc_curve(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
                   log_wandb=False):
    """Function for plotting ROC curve

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the ROC curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        fpr, tpr, _ = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        ax.plot(fpr, tpr, label=f'{labels_arr[i]} (AUC - {round(auc, 3)})')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_gmean(
            y_true, y_pred_proba, pos_label=pos_label)
        fpr, tpr, _ = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)

        ax.plot([fpr[idx]], [tpr[idx]], '-o', c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (GMean)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid()
    if log_wandb:
        wandb.log({"roc_curve": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_det_curve(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
                   log_wandb=False):
    """Function for plotting DET curve

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the DET curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        fpr, fnr, _ = det_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        auc_score = auc(fpr, fnr)
        ax.plot(norm.ppf(fpr), norm.ppf(fnr),
                label=f'{labels_arr[i]} (AUC - {round(auc_score, 3)})')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_gmean(
            y_true, y_pred_proba, pos_label=pos_label)
        fpr, fnr, _ = det_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)

        ax.plot([norm.ppf(fpr[idx])], [norm.ppf(fnr[idx])], '-o',
                c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (GMean)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('DET Curve')
    ax.legend()
    ax.grid()

    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = norm.ppf(ticks)
    tick_labels = [
        '{:.0%}'.format(s) if (100*s).is_integer() else '{:.1%}'.format(s)
        for s in ticks
    ]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)

    if log_wandb:
        wandb.log({"det_curve": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_f1(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
            log_wandb=False):
    """Function for plotting F1 curve (F1 vs threshold)
    F1 is defined as the harmonic mean between Precision and Recall
    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the F1 curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        p, r, pr_thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        f1 = 2*(p*r/(p + r))
        ax.plot(pr_thresholds, f1[:-1], label=f'{labels_arr[i]}')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_f1(
            y_true, y_pred_proba, pos_label=pos_label)
        p, r, pr_thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        f1 = 2*(p*r/(p + r))

        ax.plot([pr_thresholds[idx]], [f1[idx]], '-o', c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (F1)')

    ax.set_title('F1 vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1')
    ax.grid()
    ax.legend()
    if log_wandb:
        wandb.log({"f1": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_matthews_coeff(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
                        log_wandb=False):
    # ! Todo - MCC is returning error, and it is taking too much time. Need to debug.
    """Function for plotting MCC curve (MCC vs threshold)
    Matthews Correlation Coefficient (MCC) is different from F1 and other metrics as it takes into 
    account performance on all 4 elements of the binary confusion matrix.
    Check out https://en.wikipedia.org/wiki/Matthews_correlation_coefficient for a detailed definition.

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the MCC curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        thresholds = np.sort(y_pred_proba[:, 1])
        y_true = deepcopy(y_true)
        mcc_arr = []
        for thres in thresholds:
            y_pred = (y_pred_proba[:, 1] > thres).astype(int)
            y_true[y_true == pos_label] = 1
            y_true[y_true != pos_label] = 0

            mcc = matthews_corrcoef(
                y_true.to_numpy().astype(int), y_pred)
            mcc_arr.append(mcc)
        ax.plot(thresholds, mcc_arr, label=f'{labels_arr[i]}')

    # if plot_thres_for_idx is not None:
    #     y_true = y_true_arr[plot_thres_for_idx]
    #     y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
    #     _, idx = get_best_threshold_f1(y_true, y_pred_proba, pos_label=pos_label)
    #     p, r, pr_thresholds = precision_recall_curve(
    #         y_true, y_pred_proba[:, 1], pos_label=pos_label)
    #     f1 = 2*(p*r/(p + r))

    #     ax.plot([pr_thresholds[idx]], [f1[idx]], '-o', c=ax.lines[plot_thres_for_idx].get_color(),
    #             label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (F1)')

    ax.set_title('MCC vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('MCC')
    ax.grid()
    ax.legend()
    if log_wandb:
        wandb.log({"matthews_coeff": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_fm_index(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
                  log_wandb=False):
    """Function for plotting FM curve (Folkwes-Mallows) - FM vs Threshold.
    FM is defined as sqrt(Precision*Recall). 
    Just like F1 is the harmonic mean of P and R, FM is the geometric mean.
    Practically, its values end up being v close to the corresponding F1 values. 
    They can be interchangeably used.

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the FM curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        p, r, pr_thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        fm = np.sqrt(p*r)
        ax.plot(pr_thresholds, fm[:-1], label=f'{labels_arr[i]}')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_fowlkes_mallows(
            y_true, y_pred_proba, pos_label=pos_label)
        p, r, _ = precision_recall_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        fm = np.sqrt(p*r)

        ax.plot([pr_thresholds[idx]], [fm[idx]], '-o', c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (FM Index)')

    ax.set_title('Fowlkes–Mallows index vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Fowlkes–Mallows index')
    ax.grid()
    ax.legend()
    if log_wandb:
        wandb.log({"fm_index": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax


def plot_gmean(y_true_arr, y_pred_proba_arr, labels_arr, pos_label=None, plot_thres_for_idx=None,
               log_wandb=False):
    """Function for plotting Gmean curve (Gmean vs Threshold)
    GMean is defined as sqrt(tpr * (1 - fpr)), where tpr, fpr are the true positive, 
    and false positive rates respectively.

    Args:
        y_true_arr (list/np.array): list of all GT arrays
        y_pred_proba_arr (list/np.array): list of all predicted probabilities
        labels_arr (list/np.array): list of labels
        pos_label (str, optional): What is the label of the positive class. Defaults to 'Yes'.
        plot_thres_for_idx (int, optional): If true, best threshold (F1) is plotted 
        for the Gmean curve corresponding to this index. Defaults to None.
        log_wandb (bool, optional): If true, figure is logged to W&B. Defaults to False.

    Returns:
        plt.Figure, plt.Axes: The tuple of figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (y_true, y_pred_proba) in enumerate(zip(y_true_arr, y_pred_proba_arr)):
        fpr, tpr, roc_thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        gmean = np.sqrt(tpr * (1-fpr))
        ax.plot(roc_thresholds[1:], gmean[1:], label=f'{labels_arr[i]}')

    if plot_thres_for_idx is not None:
        y_true = y_true_arr[plot_thres_for_idx]
        y_pred_proba = y_pred_proba_arr[plot_thres_for_idx]
        _, idx = get_best_threshold_gmean(
            y_true, y_pred_proba, pos_label=pos_label)
        fpr, tpr, roc_thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=pos_label)
        gmean = np.sqrt(tpr * (1-fpr))

        ax.plot([roc_thresholds[idx]], [gmean[idx]], '-o',
                c=ax.lines[plot_thres_for_idx].get_color(),
                label=f'Best {labels_arr[plot_thres_for_idx]} Threshold (GMean)')

    ax.set_title('GMean vs threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('GMean')
    ax.grid()
    ax.legend()
    if log_wandb:
        wandb.log({"gmean": [wandb.Image(fig)]})

    plt.close(fig)
    return fig, ax
