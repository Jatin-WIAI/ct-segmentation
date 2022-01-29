import os
import pickle
import random
import time
from warnings import warn

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

import src.models as models_module
import src.utils.metrics as metrics_module
import src.utils.losses as losses_module
import src.utils.schedulers as schedulers_module
import src.viz.eval as viz_eval_module
from natsort import natsorted
from src.data.datasets.main import create_dataloader


def read_config(config_filename):
    """Read YAML config file"""
    with open(os.path.join(config_filename), "r") as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    return cfg


def setup_misc_params(cfg):
    """Setup random seeds and other torch details
    """

    # TODO: Same parameters for train.py and eval.py?
    if cfg["disable_debug_apis"]:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["cudnn_deterministic"]:
        torch.backends.cudnn.deterministic = cfg["cudnn_deterministic"]
    if cfg["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]

    return None


def setup_checkpoint_dir(cfg, args, phase):
    """Create checkpoint directory

    # ROUND2-TODO: let's make this checkpoint director way more involved. Specific to user, to model, to config name, etc.
    """

    root_dir = os.path.join(
        cfg["checkpoints_dir"], cfg["model"]["name"], args.config_name
    )

    ckpt_dir = os.path.join(root_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        if phase == "train":
            os.makedirs(ckpt_dir)
        else:
            raise FileNotFoundError("Checkpoint directory doesn't exist!")

    return ckpt_dir, root_dir


def load_checkpoints(cfg, args, model, optimizer, checkpoint_id="last"):
    print(f"> ***** Loading checkpoint from {args.config_name} *****")

    checkpoint_dir = os.path.join(
        cfg["checkpoints_dir"], cfg["model"]["name"], args.config_name, "checkpoints"
    )

    if checkpoint_id == "best":
        # Loading from best model checkpoint
        ckpt = "best_model.pth.tar"
    elif checkpoint_id == "last":
        # Loading the last saved checkpoint
        ckpt = natsorted(os.listdir(checkpoint_dir))[-1]
    else:
        # Loading from the specific checkpoint ID
        ckpt = "checkpoint-{}.pth.tar".format(checkpoint_id)

    try:
        # If loading checkpoints fails, last checkpoint is loaded by default.
        state_dict = torch.load(os.path.join(checkpoint_dir, ckpt))

    except FileNotFoundError:
        warn(
            f"{os.path.join(checkpoint_dir, ckpt)} not found. Loading from last saved checkpoint."
        )
        ckpt = natsorted(os.listdir(checkpoint_dir))[-1]
        state_dict = torch.load(os.path.join(checkpoint_dir, ckpt))

    finally:
        model.load_state_dict(state_dict["model_state_dict"])

    print(f"> ***** Resuming from checkpoint {ckpt} ** ***")

    # The below objects are don't cares during inference.
    start_epoch = state_dict["epoch"]
    metric = state_dict["metric"]
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = cfg["train"]["optimizer_params"]["lr"]

    return model, optimizer, start_epoch, metric


def init_wandb(cfg, args, root_dir, model, phase):
    # Create Wandb run name
    if phase == "inference":
        wandb_run = "{}_{}_{}".format(cfg["model"]["name"], args.config_name, phase)
    else:
        if args.run:
            wandb_run = args.run
        else:
            wandb_run = "{}_{}_{}".format(cfg["model"]["name"], args.config_name, phase)
    # Initialize
    if phase == "train":
        wandb.init(
            name=wandb_run,
            project="TB_Ultrasound",
            config=cfg,
            dir=root_dir,
            resume=args.resume,
            id=args.id,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb.init(
            name=wandb_run,
            project="TB_Ultrasound",
            config=cfg,
            dir=root_dir,
            settings=wandb.Settings(start_method="fork"),
        )
    wandb.watch(model)


def log_to_wandb(
    figures_dict,
    phase,
    train_metrics=None,
    train_loss=None,
    val_metrics=None,
    val_loss=None,
    test_metrics=None,
    test_loss=None,
    epochID=0,
):
    if phase not in ["train", "inference"]:
        raise Exception("Invalid phase! Please choose one of ['train', 'inference']")
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


def get_dataloader(cfg, phase):
    print("Creating the {} dataloader...".format(phase))

    dataloader = create_dataloader(cfg["data"], phase=phase)

    return dataloader


def initialise_objs(cfg, device, phase):
    print("Initializing the model...")
    tick = time.time()
    model_class = getattr(models_module, cfg["model"]["name"])
    model = model_class(**cfg["model_params"]).to(device)
    print("> Time to initialize model : {:.4f} sec".format(time.time() - tick))

    # Create criterion object
    criterion_class = getattr(losses_module, cfg[phase]["loss"])

    if 'weight' in cfg[phase]["loss_params"].keys():
        cfg[phase]["loss_params"]["weight"] = torch.tensor(
            cfg[phase]["loss_params"]["weight"])
    
    criterion = criterion_class(**cfg[phase]["loss_params"]).to(device)

    if phase == "train":
        # Create optimiser object
        optimizer_class = getattr(torch.optim, cfg[phase]["optimizer"])
        optimizer = optimizer_class(
            model.parameters(), **cfg[phase]["optimizer_params"]
        )
        scheduler_class = getattr(schedulers_module, cfg[phase]["lr_scheduler"])
        scheduler = scheduler_class(optimizer, **cfg[phase]["lr_scheduler_params"])
    else:
        optimizer = None
        scheduler = None

    return model, criterion, optimizer, scheduler


def evaluate(gt, preds, metric_type, cfg, phase):
    return_dict = {}
    for metric in cfg["eval"]["logging_metrics"]["{}_metrics".format(metric_type)]:
        callable_metric = getattr(metrics_module, metric)
        return_dict["{}_{}".format(phase, metric)] = callable_metric(gt, preds)

    return return_dict


def calculate_metrics(cfg, gt_labels, pred_scores, pred_labels, phase, dataloader):
    metrics_dict = {}
    if (
        cfg["task_type"] == "binary-classification"
        or cfg["task_type"] == "multiclass-classification"
    ):
        # TODO: Make scores implementation compatible for multiple classes
        metrics_dict.update(evaluate(gt_labels, pred_scores, "score", cfg, phase))
        metrics_dict.update(evaluate(gt_labels, pred_labels, "label", cfg, phase))
    elif cfg["task_type"] == "multilabel-classification":
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
    else:
        raise ValueError(
            "Support for given task_type is not yet present in the metrics module."
            + "Please choose one of - `binary-classification`, `multiclass-classification`, or `multilabel-classification`"
        )

    return metrics_dict


def create_figures(
    cfg,
    phase,
    train_gt_labels=None,
    train_scores=None,
    val_gt_labels=None,
    val_scores=None,
    test_gt_labels=None,
    test_scores=None,
    labels_dict=None,
):
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


def epoch(cfg, model, dataloader, criterion, optimizer, device, phase, scaler):
    """
    ROUND2-TODO: Generally good to pass outputs as a dictionary, makes it easier to store variables

    This function implements one epoch of training or evaluation

    Args:
        cfg (dict): Configuration file used to run the code.
        model (pytorch model): Network architecture which is being trained.
        dataloader (pytorch Dataloader): Train/ Val/ Test dataloader.
        criterion (pytorch Criterion): Loss function specified in the config file.
        optimizer (pytorch Optimizer): Optimizer specified in the config file.
        device (pytorch device): Specifies whether to run on CPU or GPU.
        phase (str): Run a training loop or inference loop. Should be one of "train" or "inference".
        scaler (scaler object): Scaler used during AMP training.

    Returns:
        (float, dict, list, list, list, list): A tuple containing the average loss, metrics computed, ground truth labels, predicted score, predicted labels, and a list containing the file paths of all data samples encountered in the epoch.
    """

    print("*" * 15 + f" || {phase} || " + "*" * 15)
    if phase == "train":
        model.train()
    else:
        model.eval()

    losses = []
    batch_times = []
    gt_labels = []
    pred_scores = []
    pred_labels = []
    image_paths = []
    symbol = "#"
    width = 40
    total = len(dataloader)

    tick = time.time()
    for batchID, (images, masks) in enumerate(dataloader):
        import pdb; pdb.set_trace()
        tock = time.time()
        labels = labels.float()
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=cfg[phase]["use_amp"]):
            with torch.set_grad_enabled(phase == "train"):
                output = model(images.to(device))
                if (
                    cfg["model"]["name"] == "inception"
                    or cfg["model"]["name"] == "googlenet"
                ):
                    output = output.logits
                loss = criterion(output, labels)

        current = batchID + 1
        percent = current / float(total)
        size = int(width * percent)
        batch_time = tock - tick
        bar = "[" + symbol * size + "." * (width - size) + "]"
        print(
            "\r Data={:.4f} s | ({:4d}/{:4d}) | {:.2f}% | Loss={:.4f} {}".format(
                batch_time, current, total, percent * 100, loss.item(), bar
            ),
            end="",
        )

        losses.append(loss.item())
        batch_times.append(batch_time)
        # Append GT Labels and Pred Scores
        gt_labels += list(labels.cpu().float().numpy())
        if phase == "train":
            scores = torch.sigmoid(output).detach().cpu().numpy()
        else:
            scores = torch.sigmoid(output).cpu().numpy()
        pred_scores += scores.tolist()

        pred_labels += (scores > 0.5).astype(float).tolist()

        if phase == "train":
            # Setting grad=0 using another method instead of optimizer.zero_grad()
            # See: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            for param in model.parameters():
                param.grad = None
            if cfg[phase]["use_amp"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        tick = time.time()

        image_paths.extend(image_path)

    average_loss = np.mean(losses)
    average_time = np.mean(batch_times)
    gt_labels = np.array(gt_labels)
    pred_scores = np.array(pred_scores)
    pred_labels = np.array(pred_labels)

    metrics_dict = calculate_metrics(
        cfg, gt_labels, pred_scores, pred_labels, phase, dataloader
    )

    bar = "[" + symbol * width + "]"
    print(
        "\rData={:.4f} s | ({:4d}/{:4d}) | 100.00% | Loss={:.4f} {}".format(
            average_time, total, total, average_loss, bar
        )
    )

    return average_loss, metrics_dict, gt_labels, pred_scores, pred_labels, image_paths


def save_model_checkpoints(
    cfg, phase, metrics, ckpt_metric, ckpt_dir, model, optimizer, epochID
):
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


def parse_image_path(path):
    # Round2-TODO: Check if the path is consistent with ms1-data

    path_items = path.split("/")
    dataset = path_items[-7]
    data = path_items[-6]
    class_label = path_items[-5]
    patient = path_items[-4]
    video = path_items[-3]
    annotator_task = path_items[-2]
    image_num = path_items[-1]

    return dataset, data, class_label, patient, video, annotator_task, image_num


def get_cached_dict(filepath):
    f = open(filepath, "rb")
    saved_obj = pickle.load(f)
    cached_dict, figures, metrics = saved_obj

    return cached_dict, figures, metrics
