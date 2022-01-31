import os

import pandas as pd
import src.utils.metrics as metrics_module
import src.viz.eval as viz_eval_module
import torch
import wandb


def log_to_wandb(figures_dict, phase, train_metrics=None, train_loss=None, val_metrics=None, val_loss=None,
                 test_metrics=None, test_loss=None, epochID=0):
    if phase not in ["train", "inference"]:
        raise Exception(
            "Invalid phase! Please choose one of ['train', 'inference']")
    new_fig_dict = convert_to_wandb_images(figures_dict)
    wandb.log(new_fig_dict, step=epochID)
    if phase == "train":
        wandb.log(train_metrics, step=epochID)
        wandb.log(val_metrics, step=epochID)
        wandb.log({"train_loss": train_loss}, step=epochID)
        wandb.log({"val_loss": val_loss}, step=epochID)
    else:
        wandb.log(test_metrics, step=epochID)
        wandb.log({"test_loss": test_loss}, step=epochID)

    return None

def evaluate(gt, preds, metric_type, cfg, phase):
    return_dict = {}
    for metric in cfg["eval"]["logging_metrics"]["{}_metrics".format(metric_type)]:
        callable_metric = getattr(metrics_module, metric)
        return_dict["{}_{}".format(phase, metric)] = callable_metric(gt, preds)

    return return_dict


def calculate_metrics(cfg, gt_dict, pred_dict, phase, dataloader):
    metrics_dict = {}
    if (cfg["task_type"] == "binary-classification"
        or cfg["task_type"] == "multiclass-classification"):
        gt_labels = gt_dict['gt_labels']
        pred_labels = pred_dict['pred_labels']
        pred_scores = pred_dict['pred_scores']
        # TODO: Make scores implementation compatible for multiple classes
        metrics_dict.update(evaluate(gt_labels, pred_scores, "score", cfg, phase))
        metrics_dict.update(evaluate(gt_labels, pred_labels, "label", cfg, phase))
    elif cfg["task_type"] == "multilabel-classification":
        gt_labels = gt_dict['gt_labels']
        pred_labels = pred_dict['pred_labels']
        pred_scores = pred_dict['pred_scores']
        name_to_code_mapping = dataloader.dataset.name_to_feature_code_mapping
        code_to_name_mapping = {v: k for k, v in name_to_code_mapping.items()}
        df_dict = {}
        for i in range(gt_labels.shape[1]):
            feature_name = code_to_name_mapping[i]
            feature_dict = evaluate(
                gt_labels[:, i], pred_scores[:, i], "score", cfg, phase
            )
            feature_dict.update(
                evaluate(gt_labels[:, i], pred_labels[:, i], "label", cfg, phase)
            )
            df_dict[feature_name] = feature_dict
            feature_dict = {f"{feature_name}_{k}": v for k, v in feature_dict.items()}
            metrics_dict.update(feature_dict)

        metrics_df = pd.DataFrame.from_dict(df_dict).T
        metrics_df.reset_index(inplace=True)
        metrics_dict.update(
            {phase + "_summary_table": wandb.Table(dataframe=metrics_df)}
        )
    elif cfg["task_type"] == "semantic-segmentation":
        gt_masks = gt_dict['gt_masks']
        pred_masks = pred_dict['pred_masks']
        metrics_dict.update(
            evaluate(gt_labels, pred_scores, "score", cfg, phase))

    else:
        raise ValueError(
            "Support for given task_type is not yet present in the metrics module."
            + "Please choose one of - `binary-classification`, `multiclass-classification`, or `multilabel-classification`"
        )

    return metrics_dict


def create_figures(cfg, phase, train_gt_labels=None, train_scores=None, val_gt_labels=None, val_scores=None, 
                   test_gt_labels=None, test_scores=None, labels_dict=None):
    """
    Plot figures to be logged to wandb.

    phase = "train" plots both the training and validation figures to allow better analysis. Phase = "inference" may be used for either standalone inference on validation set or on test set. The variable names use "test*" format since this will be create plots for a single forward pass (no training).

    Args:
        cfg (dict): Configuration file used to run the code.
        phase (str): Run a training loop or inference loop. Should be one of "train" or "inference".
        train_gt_labels (list, optional): Ground truth labels from the train set. Defaults to None.
        train_scores (list, optional): Class wise scores predicted by the model for the training set. Defaults to None.
        val_gt_labels (list, optional): Ground truth labels from the validation set. Defaults to None.
        val_scores (list, optional): Class wise scores predicted by the model for the validation set. Defaults to None.
        test_gt_labels (list, optional): Ground truth labels from the test set. Defaults to None.
        test_scores (list, optional): Class wise scores predicted by the model for the test set. Defaults to None.

    Raises:
        Exception: Incorrect phase values like "val" or "test" may lead to incorrect performance. Also ensures that case sensitive phase flags are passed.

    Returns:
        dict: A dictionary containing mappings to all figures that need to be plotted according to the config file
    """
    if phase not in ["train", "inference"]:
        raise Exception("Invalid phase! Please choose one of ['train', 'inference']")
    figures_dict = {}

    if phase == "train":
        if cfg["task_type"] == "binary-classification":
            y_true_arr = [train_gt_labels, val_gt_labels]
            y_pred_proba_arr = [train_scores, val_scores]
            labels_arr = ["Train", "Val"]
            plot_thres_for_idx = [1]
        elif cfg["task_type"] == "multilabel-classification":
            y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = [
                [] for i in range(4)
            ]
            for i in range(train_gt_labels.shape[1]):
                y_true_arr.append(train_gt_labels[:, i])
                y_pred_proba_arr.append(train_scores[:, i])
                labels_arr.append(labels_dict[i])
                plot_thres_for_idx.append(i)
    else:
        if cfg["task_type"] == "binary-classification":
            y_true_arr = [test_gt_labels]
            y_pred_proba_arr = [test_scores]
            labels_arr = ["Test"]
            plot_thres_for_idx = [0]
        elif cfg["task_type"] == "multilabel-classification":
            y_true_arr, y_pred_proba_arr, labels_arr, plot_thres_for_idx = [
                [] for i in range(4)
            ]
            for i in range(test_gt_labels.shape[1]):
                y_true_arr.append(test_gt_labels[:, i])
                y_pred_proba_arr.append(test_scores[:, i])
                labels_arr.append(labels_dict[i])
                plot_thres_for_idx.append(i)
    for func in cfg["viz"]["eval"]:
        if func == "plot_pr_curve":
            fig, _ = getattr(viz_eval_module, func)(
                y_true_arr,
                y_pred_proba_arr,
                labels_arr,
                plot_thres_for_idx=plot_thres_for_idx,
                pos_label=1,
                plot_prevalance_for_idx=plot_thres_for_idx,
            )
        else:
            fig, _ = getattr(viz_eval_module, func)(
                y_true_arr,
                y_pred_proba_arr,
                labels_arr,
                plot_thres_for_idx=plot_thres_for_idx,
                pos_label=1,
            )
        figures_dict[func] = fig

    return figures_dict


def convert_to_wandb_images(figures_dict):
    new_fig_dict = dict()
    for fig in figures_dict:
        new_fig_dict[fig] = [wandb.Image(figures_dict[fig])]

    return new_fig_dict


def save_model_checkpoints(cfg, phase, metrics, ckpt_metric, ckpt_dir, model, optimizer, epochID):
    state_dict = {
        "epoch": epochID,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": ckpt_metric,
    }

    metric_key = cfg["eval"]["ckpt_metric"]
    if metrics[metric_key] > ckpt_metric:
        ckpt_metric = metrics[metric_key]
        best_model_path = os.path.join(ckpt_dir, "best_model.pth.tar")
        torch.save(state_dict, best_model_path)

    model_name = "checkpoint-{}.pth.tar".format(epochID)
    model_path = os.path.join(ckpt_dir, model_name)
    torch.save(state_dict, model_path)
