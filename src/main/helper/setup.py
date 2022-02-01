import copy
import os
import pickle
import random
import time
from warnings import warn

import numpy as np
import src.models as models_module
import src.utils.losses as losses_module
import src.utils.optimizers as optimizers_module
import src.utils.schedulers as schedulers_module
import torch
import wandb
import yaml
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
        run_name = "{}_{}_{}_{}".format(cfg["data"]["dataset"], cfg["model"]["name"],
                                        args.config_name, phase)
    else:
        if args.run:
            run_name = args.run
        else:
            run_name = "{}_{}_{}_{}".format(cfg["data"]["dataset"], cfg["model"]["name"],
                                            args.config_name, phase)
    # Initialize
    wandb_args = copy.deepcopy(cfg['wandb'])
    if wandb_args['name'] is None:
        wandb_args['name'] = run_name
    wandb_args.update({
        'config': cfg,
        'dir': root_dir,
        'settings': wandb.Settings(start_method="fork"),
    })
    if phase == "train":
        wandb_args.update({
            'resume': args.resume,
            'id': args.id,
        })
    wandb.init(**wandb_args)
    wandb.watch(model)

def get_dataloader(cfg, phase):
    print("Creating the {} dataloader...".format(phase))

    dataloader = create_dataloader(cfg["data"], phase=phase)

    return dataloader


def initialise_objs(cfg, device, phase):
    print("Initializing the model...")
    tick = time.time()
    model_class = getattr(models_module, cfg["model"]["name"])
    model = model_class(**cfg["model"]["params"]).to(device)
    print("> Time to initialize model : {:.4f} sec".format(time.time() - tick))

    # Create criterion object
    criterion_class = getattr(losses_module, cfg[phase]["loss"])

    if 'weight' in cfg[phase]["loss_params"].keys():
        cfg[phase]["loss_params"]["weight"] = torch.tensor(
            cfg[phase]["loss_params"]["weight"])
    
    criterion = criterion_class(**cfg[phase]["loss_params"]).to(device)

    if phase == "train":
        # Create optimiser object
        optimizer_class = getattr(optimizers_module, cfg[phase]["optimizer"])
        optimizer = optimizer_class(
            model.parameters(), **cfg[phase]["optimizer_params"]
        )
        scheduler_class = getattr(schedulers_module, cfg[phase]["lr_scheduler"])
        scheduler = scheduler_class(optimizer, **cfg[phase]["lr_scheduler_params"])
    else:
        optimizer = None
        scheduler = None

    return model, criterion, optimizer, scheduler


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
