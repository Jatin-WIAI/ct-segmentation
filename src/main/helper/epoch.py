
import time

import numpy as np
import torch
from .postprocess import calculate_metrics


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
    gt_masks = []
    pred_masks = []
    symbol = "#"
    width = 40
    total = len(dataloader)

    tick = time.time()
    for batchID, (images, masks) in enumerate(dataloader):
        tock = time.time()
        masks = masks.to(device)

        with torch.cuda.amp.autocast(enabled=cfg[phase]["use_amp"]):
            with torch.set_grad_enabled(phase == "train"):
                output = model(images.to(device))
                if (
                    cfg["model"]["name"] == "inception"
                    or cfg["model"]["name"] == "googlenet"
                ):
                    output = output.logits
                loss = criterion(output, masks)

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
        # Append GT Masks and Pred Masks
        gt_masks += masks
        pred_masks += output

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


    average_loss = np.mean(losses)
    average_time = np.mean(batch_times)
    gt_masks = torch.cat(gt_masks, 0)
    pred_masks = torch.cat(pred_masks, 0)

    gt_dict = {
        'gt_masks': gt_masks
    }
    pred_dict = {
        'pred_masks': pred_masks
    }

    metrics_dict = calculate_metrics(cfg, gt_dict, pred_dict, phase, dataloader)

    bar = "[" + symbol * width + "]"
    print(
        "\rData={:.4f} s | ({:4d}/{:4d}) | 100.00% | Loss={:.4f} {}".format(
            average_time, total, total, average_loss, bar
        )
    )

    return average_loss, metrics_dict, gt_masks, pred_masks
