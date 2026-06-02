"""
Transfer Learning for object detection on HDB features using satellite imagery and street view images.

Author: AxeWhy
Date: 01/06/2026
Last Updated: 04/06/2026

Description:
This script implements transfer learning for object detection on HDB features using satellite imagery
and street view images. It utilizes the Faster R-CNN model with a ResNet-50 backbone, allowing for 
training on a custom COCO-style dataset. The script supports training, validation, and testing phases, 
with options for data augmentation, checkpointing, and performance evaluation.

Default values:
- Data root: ../Labelled_Dataset
- Output root: ./runs
- Train/valid/test splits: train/valid/test

Usage CLI:
To train a model with default settings:
python Transfer_Learning.py --run-name my_experiment

To resume training from a checkpoint:
python Transfer_Learning.py --run-name my_experiment --resume path/to/model_last.pth

To evaluate a trained model on the test set:
python Transfer_Learning.py --run-name my_experiment --resume path/to/model_best.pth --test-only

For more options and configurations, use:
python Transfer_Learning.py --help

Note:
To resume training, ensure the optimizer param group keys match the checkpoint.
(i.e., same --trainable-backbone-layers, --min-size, --max-size, and pretrained settings as the original run).

Dependent Libraries:
- torch
- torchvision
- PIL
- matplotlib (optional, for plotting)
- OpenCV (optional, for visualization)

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, nms
from torchvision.transforms import functional as F


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

def get_plotting_backend():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_root = script_dir.parent / "Labelled_Dataset"
    default_output_root = script_dir / "runs"

    parser = argparse.ArgumentParser(description="Train and evaluate Faster R-CNN on a COCO-style dataset.")
    parser.add_argument("--data-root", type=str, default=str(default_data_root), help="Root directory containing train/valid/test subdirectories with JSON annotations and images.")
    parser.add_argument("--train-dir", type=str, default="train", help="Name of the training subdirectory.")
    parser.add_argument("--valid-dir", type=str, default="valid", help="Name of the validation subdirectory.")
    parser.add_argument("--test-dir", type=str, default="test", help="Name of the test subdirectory.")
    parser.add_argument("--annotations-filename", type=str, default="result.json", help="Name of the JSON annotations file.")
    parser.add_argument("--images-dirname", type=str, default="images", help="Name of the images subdirectory.")
    parser.add_argument("--output-root", type=str, default=str(default_output_root), help="Root directory for saving outputs.")
    parser.add_argument("--run-name", type=str, default="", help="Name of the run for logging and saving purposes.")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4, help="Number of worker processes for data loading.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate for the optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay for the optimizer.")
    parser.add_argument("--lr-step-size", type=int, default=3, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Gamma for the learning rate scheduler.")
    parser.add_argument("--max-grad-norm", type=float, default=10.0, help="Maximum norm for gradient clipping.")

    parser.add_argument("--trainable-backbone-layers", type=int, default=3, help="Number of trainable backbone layers.")
    parser.add_argument("--min-size", type=int, default=None, help="Minimum image size.")
    parser.add_argument("--max-size", type=int, default=None, help="Maximum image size.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use pretrained weights.")

    parser.add_argument("--score-threshold", type=float, default=0.5, help="Minimum score threshold for detections.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Minimum IoU threshold for detections.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Probability of horizontal flip augmentation.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--print-freq", type=int, default=20, help="Frequency of printing training progress.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of samples for debugging.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing images during training.")

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from.")
    parser.add_argument("--test-only", action="store_true", help="Run evaluation only.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation after training.")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs.")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_id = args.run_name.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / run_id
    setup_logging(output_dir)

    plot_backend = get_plotting_backend()
    graphical_dir = output_dir / "graphical_results"
    if plot_backend is None:
        logging.warning("matplotlib not available; skipping graphical results.")

    logging.info("Starting run %s", run_id)
    logging.info("Arguments: %s", args)

    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        logging.info("Using CUDA device: %s", device)

    dataset_info, train_loader, valid_loader, test_loader = build_dataloaders(args)
    num_classes = len(dataset_info.label_to_name) + 1
    logging.info("Detected %d classes (excluding background).", num_classes - 1)

    model = create_model(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.to(device)

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_metric = 0.0
    start_epoch = 1
    history = {
        "train_loss": [],
        "val_loss": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mean_iou": [],
    }

    if args.resume:
        if args.test_only:
            load_checkpoint(Path(args.resume), model)
            logging.info("Loaded model weights from %s", args.resume)
        else:
            logging.warning("--resume flag is set without --test-only; Please ensure the optimizer param group keys match the checkpoint (e.g., same --trainable-backbone-layers, --min-size, --max-size, and pretrained settings as the original run).")
            checkpoint = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_metric = checkpoint.get("best_metric", 0.0)
            logging.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    if args.test_only:
        if not args.resume:
            logging.warning("--test-only set without --resume; evaluating an untrained model.")
        if args.resume:
            logging.info("Evaluating on test split using %s", args.resume)
        metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )
        logging.info("Test metrics: %s", metrics)
        visualize_predictions(
            model=model,
            data_loader=test_loader,
            dataset=test_loader.dataset,
            device=device,
            output_dir=output_dir / "test_prediction",
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            label_to_name=dataset_info.label_to_name,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler,
            max_norm=args.max_grad_norm,
            print_freq=args.print_freq,
        )

        val_loss = compute_validation_loss(
            model=model,
            data_loader=valid_loader,
            device=device,
            use_amp=args.amp,
        )

        metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )

        scheduler.step()

        logging.info(
            "Epoch %d | Train loss %.4f | Val loss %.4f | Val precision %.4f | Val recall %.4f | Val F1 %.4f | Val mIoU %.4f",
            epoch,
            train_loss,
            val_loss,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["mean_iou"],
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])
        history["mean_iou"].append(metrics["mean_iou"])

        if plot_backend is not None:
            save_training_plots(history, graphical_dir, plot_backend)

        is_best = metrics["f1"] > best_metric
        if is_best:
            best_metric = metrics["f1"]
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            dataset_info=dataset_info,
            is_best=is_best,
        )

        if args.save_every and epoch % args.save_every == 0:
            extra_path = output_dir / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), extra_path)

    logging.info("Training complete. Best F1: %.4f", best_metric)

    if args.skip_test:
        logging.info("Skipping test evaluation as per --skip-test flag.")
        return

    best_checkpoint = output_dir / "model_best.pth"
    if best_checkpoint.exists():
        load_checkpoint(best_checkpoint, model)
        logging.info("Loaded best model from %s for test evaluation.", best_checkpoint)
    else:
        logging.warning("Best checkpoint not found at %s; evaluating last epoch model.", best_checkpoint)

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    logging.info("Test metrics: %s", test_metrics)
    visualize_predictions(
        model=model,
        data_loader=test_loader,
        dataset=test_loader.dataset,
        device=device,
        output_dir=output_dir / "test_prediction",
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        label_to_name=dataset_info.label_to_name,
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
