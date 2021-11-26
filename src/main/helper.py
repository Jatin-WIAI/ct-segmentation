import os
import random
import time
from warnings import warn

import numpy as np
import torch
import wandb
import yaml
from natsort import natsorted
import segmentation_models_pytorch as smp

import src.models as models_module
import src.utils.metrics as metrics_module
import src.viz.eval as viz_eval_module
import src.utils.losses as losses_module
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
        wandb_run = "{}_{}_{}".format(
            cfg["model"]["name"], args.config_name, phase)
    else:
        if args.run:
            wandb_run = args.run
        else:
            wandb_run = "{}_{}_{}".format(
                cfg["model"]["name"], args.config_name, phase)
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
        raise Exception(
            "Invalid phase! Please choose one of ['train', 'inference']")
    wandb.log(figures_dict, step=epochID)
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
    # Assumption here that the models that will be called will be from the segmentation_models_pytorch module
    model_class = getattr(smp, cfg["model"])
    model = model_class(**cfg["model_params"]).to(device)
    print("> Time to initialize model : {:.4f} sec".format(time.time() - tick))

    # Create criterion object
    criterion_class = getattr(losses_module, cfg[phase]["loss"])
    if cfg[phase]["loss"] == 'CrossEntropyLoss':
        weight = torch.tensor(cfg[phase]["loss_params"]["weight"])
        criterion = criterion_class(weight=weight).to(device)
    else:
        criterion = criterion_class(**cfg[phase]["loss_params"]).to(device)

    if phase == "train":
        # Create optimiser object
        optimizer_class = getattr(torch.optim, cfg[phase]["optimizer"])
        optimizer = optimizer_class(
            model.parameters(), **cfg[phase]["optimizer_params"]
        )
    else:
        optimizer = None

    return model, criterion, optimizer


def evaluate(gt, preds, metric_type, cfg, metrics_dict, phase):
    for metric in cfg["eval"]["logging_metrics"]["{}_metrics".format(metric_type)]:
        callable_metric = getattr(metrics_module, metric)
        metrics_dict["{}_{}".format(phase, metric)
                     ] = callable_metric(gt, preds)

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
        raise Exception(
            "Invalid phase! Please choose one of ['train', 'inference']")
    figures_dict = {}
    if phase == "train":
        y_true_arr = [train_gt_labels, val_gt_labels]
        y_pred_proba_arr = [train_scores, val_scores]
        labels_arr = ["Train", "Val"]
        plot_thres_for_idx = 1
    else:
        y_true_arr = [test_gt_labels]
        y_pred_proba_arr = [test_scores]
        labels_arr = ["Test"]
        plot_thres_for_idx = 0
    for func in cfg["viz"]["eval"]:
        if func == "plot_pr_curve":
            fig, _ = getattr(viz_eval_module, func)(
                y_true_arr,
                y_pred_proba_arr,
                labels_arr,
                plot_thres_for_idx=plot_thres_for_idx,
                pos_label=1,
                plot_prevalance_for_idx=0,
            )
        else:
            fig, _ = getattr(viz_eval_module, func)(
                y_true_arr,
                y_pred_proba_arr,
                labels_arr,
                plot_thres_for_idx=plot_thres_for_idx,
                pos_label=1,
            )
        figures_dict[func] = [wandb.Image(fig)]

    return figures_dict


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
    for batchID, (images, labels, image_path) in enumerate(dataloader):
        tock = time.time()
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
            scores = (torch.softmax(output, -1)).detach().cpu().numpy()
        else:
            scores = (torch.softmax(output, -1)).cpu().numpy()
        pred_scores += scores.tolist()

        pred_label = torch.argmax(torch.softmax(output, -1), -1).cpu().numpy()
        pred_labels += pred_label.tolist()

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

    metrics_dict = dict()

    # TODO: Make scores implementation compatible for multiple classes
    metrics_dict = evaluate(
        gt_labels, pred_scores[:, 1], "score", cfg, metrics_dict, phase
    )
    metrics_dict = evaluate(gt_labels, pred_labels,
                            "label", cfg, metrics_dict, phase)

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

    metric_key = "{}_{}".format(phase, cfg["eval"]["ckpt_metric"])
    if metrics[metric_key] > ckpt_metric:
        ckpt_metric = metrics[metric_key]
        best_model_path = os.path.join(ckpt_dir, "best_model.pth.tar")
        torch.save(state_dict, best_model_path)

    model_name = "checkpoint-{}.pth.tar".format(epochID)
    model_path = os.path.join(ckpt_dir, model_name)
    torch.save(state_dict, model_path)
