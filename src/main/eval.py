# This script compute metrics and cache predictions on val/test set
# Flow: parse args -> get predictions -> compute metrics -> cache results
# Usage:
# PYTHONPATH='/home/users/arsh/ultrasound' \
# CUDA_VISIBLE_DEVICES=1 \
# taskset --cpu-list 10-19 \
# python src/main/eval.py --config va5.yml


import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

# pickle module cannot serialize lambda functions
import dill as pickle
import src.utils.constants as constants
from helper import (
    create_figures,
    epoch,
    get_dataloader,
    init_wandb,
    initialise_objs,
    load_checkpoints,
    log_to_wandb,
    parse_image_path,
    read_config,
    setup_checkpoint_dir,
    setup_misc_params,
)


def init_preds_dict():
    return {
        "image_paths": [],
        "gt": [],
        "pred_label": [],
        "pred_prob": [],
    }


def cache_predictions(
    gt,
    pred_prob,
    pred_labels,
    image_paths,
    eval_mode,
    CKPT_ROOT,
    figures_dict=None,
    metrics=None,
):
    """

    This saves prediction outputs in a particular structure at /path/to/checkpoint/root/cache.
    These caches can be used in notebook for result analysis

    Output: The following files are cached.
    - `pred_dict` : same info structured with patientID/videoID
        ```
        pred_dict[patient][video] = {
                'image_paths' : [],   # list of all images corresponding to the patient and video ID
                'gt' : [],            # list of gt corresponding to the patient and video ID
                'pred_label' : [],    # list of pred label corresponding to the patient and video ID
                'pred_prob' : [],     # list of pred prob corresponding to the patient and video ID
            }
        ```
    """
    cache_dir = os.path.join(CKPT_ROOT, "cache_copy")  # Results cached in this dir
    os.makedirs(cache_dir, exist_ok=True)

    print("> Collating all predictions...")

    pred_dict = defaultdict(lambda: defaultdict(init_preds_dict))

    for i, image_path in enumerate(image_paths):
        _, _, _, patient, video, _, _ = parse_image_path(image_path)

        # ROUND2-TODO: Extend to frame-level and video-level probabilities.
        pred_dict[patient][video]["image_paths"].append(image_path)
        pred_dict[patient][video]["gt"].append(gt[i].item())
        pred_dict[patient][video]["pred_label"].append(pred_labels[i].item())
        pred_dict[patient][video]["pred_prob"].append(
            pred_prob[i][pred_labels[i]].item()
        )

    save_obj = [pred_dict, figures_dict, metrics]

    print("> Saving all files...")

    clf_dict_file = os.path.join(cache_dir, f"{eval_mode}_pred_dict.pkl")
    with open(clf_dict_file, "wb") as f:
        pickle.dump(save_obj, f)


def cache_xray_predictions(
    gt,
    pred_prob,
    pred_labels,
    image_paths,
    eval_mode,
    CKPT_ROOT,
    figures_dict=None,
    metrics=None,
):
    cache_dir = os.path.join(CKPT_ROOT, "cache_copy")  # Results cached in this dir
    os.makedirs(cache_dir, exist_ok=True)

    print("> Collating all predictions...")

    pred_dict = dict()

    for i, image_path in enumerate(image_paths):
        image_pred_dict = dict()
        image_pred_dict["gt"] = gt[i]
        image_pred_dict["pred_label"] = pred_labels[i]
        image_pred_dict["pred_prob"] = pred_prob[i]
        image_name = image_path.split("/")[-1]
        pred_dict[image_name] = image_pred_dict

    save_obj = [pred_dict, figures_dict, metrics]

    print("> Saving all files...")

    clf_dict_file = os.path.join(cache_dir, f"{eval_mode}_pred_dict.pkl")
    with open(clf_dict_file, "wb") as f:
        pickle.dump(save_obj, f)


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

    print("Creating the dataloaders...")
    test_dataloader = get_dataloader(cfg, phase=args.eval_mode)

    print("Initializing the model...")
    # Initializing model and criterion

    model, criterion, _, _ = initialise_objs(cfg, device, phase="inference")
    ckpt_dir, root_dir = setup_checkpoint_dir(cfg, args, phase="inference")

    print(f"Loading from checkpoints at {ckpt_dir}")

    inference_checkpoint = cfg["inference"]["inference_checkpoint"]
    model, _, _, _ = load_checkpoints(
        cfg, args, model, None, checkpoint_id=inference_checkpoint
    )

    # Initialise W&B
    if args.wandb:
        init_wandb(cfg, args, root_dir, model, "inference")

    image_path_list = []

    print("-" * 50)
    print("Evaluating...")

    start_time = time.time()
    (
        test_loss,
        test_metrics,
        test_gt_labels,
        test_scores,
        test_pred_labels,
        image_paths,
    ) = epoch(
        cfg,
        model,
        test_dataloader,
        criterion,
        None,
        device,
        phase="inference",
        scaler=scaler,
    )
    test_end_time = time.time()
    image_path_list.extend(image_paths)
    print("> Time to run full epoch : {:.4f} sec".format(test_end_time - start_time))

    labels_dict = test_dataloader.dataset.name_to_feature_code_mapping
    labels_dict = {v: k for k, v in labels_dict.items()}
    figures_dict = create_figures(
        cfg,
        "inference",
        test_gt_labels=test_gt_labels,
        test_scores=test_scores,
        labels_dict=labels_dict,
    )

    start_time = time.time()
    if args.wandb:
        log_to_wandb(
            figures_dict,
            "inference",
            test_metrics=test_metrics,
            test_loss=test_loss,
            epochID=0,
        )
    print("> Time to log to W&B : {:.4f} sec".format(time.time() - start_time))

    print("Evaluation Completed!")

    cache_xray_predictions(
        test_gt_labels,
        test_scores,
        test_pred_labels,
        image_paths,
        args.eval_mode,
        root_dir,
        figures_dict,
        test_metrics,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="TB USG model evaluation")
    parser.add_argument(
        "-c", "--config", type=str, help="Config version to be used for evaluation.",
    )
    parser.add_argument("--wandb", action="store_true", help="Use Wandb")
    parser.add_argument(
        "--eval_mode", type=str, default="test", help="Eval mode val/test"
    )
    args = parser.parse_args()
    args.config_name = os.path.splitext(args.config)[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
