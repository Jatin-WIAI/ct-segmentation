# Training code for TB Ultrasound classification task
# Usage: See `train.sh` for a sample run command
# NOTE:
# - Default config version is `v0` for quick testing
# - Do not run the code with existing versions. This might
#   overwrite the exisiting checkpoints.

# Usage:
# PYTHONPATH='/home/users/arsh/ultrasound' \
# CUDA_VISIBLE_DEVICES=1 \
# taskset --cpu-list 10-19 \
# python src/main/train.py --config va5.yml \
# [--wandb] [--resume]

import argparse
import os
import sys
import time
from pprint import pprint

import torch

sys.path.append('../../')

import src.utils.constants as constants
from src.main.helper import (create_figures, epoch, get_dataloader, init_wandb,
                             initialise_objs, load_checkpoints, log_to_wandb,
                             read_config, save_model_checkpoints,
                             setup_checkpoint_dir, setup_misc_params)

sys.path.append("../../")


def main(args):
    # Setting up necessary parameters
    print("Setting up parameters...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scaler = torch.cuda.amp.GradScaler()

    # Load config
    cfg_filename = os.path.join(constants.CONFIG_DIR, args.config)
    cfg = read_config(cfg_filename)

    # Setup parameters
    setup_misc_params(cfg)

    # Set checkpoint directory
    ckpt_dir, root_dir = setup_checkpoint_dir(cfg, args, phase="train")
    print(f"Storing checkpoints at {ckpt_dir}")

    # Creating dataloaders
    train_dataloader = get_dataloader(cfg, phase="train")
    # Using phase == 'val' only for dataloaders. phase == 'inference' used everywhere else
    val_dataloader = get_dataloader(cfg, phase="val")

    # Initializing model, loss function and optimizers
    model, criterion, optimizer, scheduler = initialise_objs(cfg, device, phase="train")

    # If resume flag, load the checkpoints
    if args.resume:
        resume_checkpoint = cfg["train"]["resume_checkpoint"]
        model, optimizer, start_epoch, ckpt_metric = load_checkpoints(
            cfg, args, model, optimizer, resume_checkpoint
        )
    else:
        start_epoch = 0
        ckpt_metric = 0

    # Prepare environment for training
    print("Preparing for training...")
    # Initialise W&B
    if args.wandb:
        init_wandb(cfg, args, root_dir, model, "train")

    # Run and log one validation epoch before beginning training
    # VAL EPOCH
    val_start_time = time.time()
    val_loss, val_metrics, val_gt_labels, val_scores, val_pred_labels, _ = epoch(
        cfg, model, val_dataloader, criterion, None, device, phase="inference", scaler=scaler)
    val_end_time = time.time()
    print("> Time to run full val epoch : {:.4f} sec".format(
        val_end_time - val_start_time))
    
    log_start_time = time.time()
    if args.wandb:
        log_to_wandb(
            {}, "train", {}, {}, val_metrics, val_loss, epochID=0,
        )
    print("> Time to log to W&B : {:.4f} sec".format(time.time() - log_start_time))

    # Run training
    for epochID in range(start_epoch, start_epoch + cfg["train"]["num_epochs"]):
        print("-" * 50)
        print("Training for epoch {}/{}".format(epochID + 1, cfg["train"]["num_epochs"]))

        # TRAIN EPOCH
        start_time = time.time()
        (train_loss, train_metrics, train_gt_labels, train_scores, train_pred_labels, _) = epoch(
            cfg, model, train_dataloader, criterion, optimizer, device, phase="train", scaler=scaler)
        train_end_time = time.time()
        print("> Time to run full train epoch : {:.4f} sec".format(
            train_end_time - start_time))

        # VAL EPOCH
        val_loss, val_metrics, val_gt_labels, val_scores, val_pred_labels, _ = epoch(
            cfg, model, val_dataloader, criterion, None, device, phase="inference", scaler=scaler)
        val_end_time = time.time()
        print("> Time to run full val epoch : {:.4f} sec".format(
            val_end_time - train_end_time))
        print("> Time to run full epoch : {:.4f} sec".format(val_end_time - start_time))

        scheduler.step()

        # LOGGING AND PLOTTING
        labels_dict = train_dataloader.dataset.name_to_feature_code_mapping
        labels_dict = {v: k for k, v in labels_dict.items()}
        figures_dict = create_figures(cfg, "train", train_gt_labels, train_scores, 
                                      val_gt_labels, val_scores, labels_dict=labels_dict)

        start_time = time.time()
        pprint(train_metrics)
        pprint(val_metrics)
        if args.wandb:
            log_to_wandb(figures_dict, "train", train_metrics, train_loss, val_metrics, 
                         val_loss, epochID=epochID + 1,)
        print("> Time to log to W&B : {:.4f} sec".format(time.time() - start_time))

        # SAVE CHECKPOINTS
        save_model_checkpoints(cfg, "inference", val_metrics, ckpt_metric, 
                               ckpt_dir, model, optimizer, epochID + 1,)

    print("Training Completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="TB USG model training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="v16.yml",
        help="Config version to use for training",
    )
    parser.add_argument("--wandb", action="store_true", help="Use Wandb or not")
    parser.add_argument("--run", type=str, help="Wandb experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume experiment")
    parser.add_argument(
        "--id", type=str, help="Wandb experiment ID to resume experiment"
    )
    args = parser.parse_args()
    args.config_name = os.path.splitext(args.config)[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
