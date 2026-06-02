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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch: Iterable[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List, List]:
    images, targets = zip(*batch)
    return list(images), list(targets)


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        if random.random() < self.probability:
            image = F.hflip(image)
            width = image.size[0]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def get_transforms(train: bool, hflip_prob: float) -> Compose:
    transforms: List = [ToTensor()]
    if train and hflip_prob > 0:
        transforms.insert(0, RandomHorizontalFlip(hflip_prob))
    return Compose(transforms)


@dataclass
class DatasetInfo:
    category_id_to_label: Dict[int, int]
    label_to_name: Dict[int, str]


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        images_dir: Path,
        transforms: Optional[Compose] = None,
        category_id_to_label: Optional[Dict[int, int]] = None,
        label_to_name: Optional[Dict[int, str]] = None,
        verify_images: bool = True,
        skip_missing: bool = False,
        limit: int = 0,
    ) -> None:
        self.json_path = json_path
        self.images_dir = images_dir
        self.transforms = transforms
        self.verify_images = verify_images
        self.skip_missing = skip_missing

        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        images = data.get("images", [])
        categories = data.get("categories", [])
        annotations = data.get("annotations", [])

        if category_id_to_label is None:
            if categories:
                sorted_categories = sorted(categories, key=lambda item: item["id"])
                category_id_to_label = {cat["id"]: idx + 1 for idx, cat in enumerate(sorted_categories)}
                label_to_name = {idx + 1: cat.get("name", str(cat["id"])) for idx, cat in enumerate(sorted_categories)}
            else:
                unique_ids = sorted({ann["category_id"] for ann in annotations})
                category_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(unique_ids)}
                label_to_name = {idx + 1: str(cat_id) for idx, cat_id in enumerate(unique_ids)}
        else:
            if label_to_name is None:
                label_to_name = {label: str(cat_id) for cat_id, label in category_id_to_label.items()}

        self.dataset_info = DatasetInfo(
            category_id_to_label=category_id_to_label,
            label_to_name=label_to_name,
        )

        self._annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for ann in annotations:
            self._annotations_by_image[ann["image_id"]].append(ann)

        json_dir = json_path.parent
        image_records = []
        for image_info in images:
            resolved = self._resolve_image_path(json_dir, image_info)
            if resolved is None:
                message = f"Missing image for {image_info.get('file_name', 'unknown')} in {json_path}"
                if skip_missing:
                    logging.warning(message)
                    continue
                raise FileNotFoundError(message)
            image_records.append({"info": image_info, "path": resolved})

        if limit > 0:
            image_records = image_records[:limit]

        self._image_records = image_records
        self._image_id_to_path = {
            record["info"]["id"]: record["path"]
            for record in image_records
            if "id" in record["info"]
        }

    def _resolve_image_path(self, json_dir: Path, image_info: Dict) -> Optional[Path]:
        file_name = image_info.get("file_name", "")
        if not file_name:
            return None

        candidate_paths = []
        file_path = Path(file_name)
        if file_path.is_absolute():
            candidate_paths.append(file_path)
        else:
            candidate_paths.append(json_dir / file_path)
            candidate_paths.append(self.images_dir / file_path)

        candidate_paths.append(self.images_dir / file_path.name)

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        if not self.verify_images:
            return candidate_paths[-1]
        return None

    def __len__(self) -> int:
        return len(self._image_records)

    def get_image_path(self, image_id: int) -> Optional[Path]:
        return self._image_id_to_path.get(image_id)

    def __getitem__(self, idx: int):
        record = self._image_records[idx]
        image_info = record["info"]
        image_path = record["path"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        annotations = self._annotations_by_image.get(image_info["id"], [])

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annotations:
            category_id = ann.get("category_id")
            if category_id not in self.dataset_info.category_id_to_label:
                logging.warning("Unknown category_id=%s in %s", category_id, self.json_path)
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = bbox
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.dataset_info.category_id_to_label[category_id])
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )
        areas_tensor = (
            torch.tensor(areas, dtype=torch.float32)
            if areas
            else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd_tensor = (
            torch.tensor(iscrowd, dtype=torch.int64)
            if iscrowd
            else torch.zeros((0,), dtype=torch.int64)
        )

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_info["id"]]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def create_model(
    num_classes: int,
    pretrained: bool,
    trainable_backbone_layers: int,
    min_size: Optional[int],
    max_size: Optional[int],
) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
    if min_size is not None:
        kwargs["min_size"] = min_size
    if max_size is not None:
        kwargs["max_size"] = max_size
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_norm: float,
    print_freq: int,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            raise ValueError(f"Non-finite loss detected: {losses.item():.6f}")

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        running_loss += losses.item()
        if (step + 1) % print_freq == 0:
            avg_loss = running_loss / (step + 1)
            logging.info("Epoch %d | Step %d/%d | Loss %.4f", epoch, step + 1, len(data_loader), avg_loss)

    return running_loss / max(1, len(data_loader))

 
def compute_validation_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    module_training_states = [(module, module.training) for module in model.modules()]
    model.train()

    eval_only_modules = (
        nn.modules.batchnorm._BatchNorm,
        nn.modules.dropout._DropoutNd,
    )

    def set_eval_for_validation(module: nn.Module) -> None:
        if isinstance(module, eval_only_modules):
            module.eval()

    model.apply(set_eval_for_validation)
    running_loss = 0.0
    amp_enabled = use_amp and device.type == "cuda"

    try:
        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses):
                    raise ValueError(f"Non-finite loss detected: {losses.item():.6f}")

                running_loss += losses.item()
    finally:
        for module, was_training in module_training_states:
            module.train(was_training)

    return running_loss / max(1, len(data_loader))


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    nms_iou_threshold: float,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    sum_iou = 0.0
    match_count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                pred_scores = output["scores"].cpu()
                keep = pred_scores >= score_threshold
                pred_boxes = output["boxes"].cpu()[keep]
                pred_labels = output["labels"].cpu()[keep]
                pred_scores = pred_scores[keep]

                pred_boxes, pred_scores, pred_labels = apply_nms(
                    pred_boxes,
                    pred_scores,
                    pred_labels,
                    nms_iou_threshold,
                )

                label_set = set(gt_labels.tolist()) | set(pred_labels.tolist())
                for label in label_set:
                    gt_mask = gt_labels == label
                    pred_mask = pred_labels == label
                    gt = gt_boxes[gt_mask]
                    pred = pred_boxes[pred_mask]
                    pred_score_label = pred_scores[pred_mask]

                    if len(gt) == 0 and len(pred) == 0:
                        continue
                    if len(gt) == 0:
                        fp += len(pred)
                        continue
                    if len(pred) == 0:
                        fn += len(gt)
                        continue

                    order = torch.argsort(pred_score_label, descending=True)
                    pred = pred[order]
                    ious = box_iou(pred, gt)
                    matched_gt = set()

                    for pred_idx in range(pred.shape[0]):
                        iou_row = ious[pred_idx]
                        max_iou, gt_idx = torch.max(iou_row, dim=0)
                        if max_iou >= iou_threshold and int(gt_idx) not in matched_gt:
                            tp += 1
                            matched_gt.add(int(gt_idx))
                            sum_iou += float(max_iou)
                            match_count += 1
                        else:
                            fp += 1

                    fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = sum_iou / match_count if match_count > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def get_plotting_backend():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def save_training_plots(
    history: Dict[str, List[float]],
    output_dir: Path,
    plt,
) -> None:
    if not history.get("train_loss"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    metric_keys = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("mean_iou", "Mean IoU"),
    ]
    if any(history.get(key) for key, _ in metric_keys):
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, label in metric_keys:
            values = history.get(key, [])
            if values:
                ax.plot(epochs[: len(values)], values, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Validation Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "metrics_curve.png", dpi=200)
        plt.close(fig)


def _color_from_label(label: int) -> Tuple[int, int, int]:
    return (
        int((37 * label) % 255),
        int((17 * label) % 255),
        int((29 * label) % 255),
    )


def visualize_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    dataset: CocoDetectionDataset,
    device: torch.device,
    output_dir: Path,
    score_threshold: float,
    nms_iou_threshold: float,
    label_to_name: Dict[int, str],
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for prediction visualization.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id_tensor = target.get("image_id")
                image_id = int(image_id_tensor.item()) if image_id_tensor is not None else -1
                image_path = dataset.get_image_path(image_id)
                if image_path is None:
                    logging.warning("Missing image path for image_id=%s; skipping visualization.", image_id)
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    logging.warning("Failed to read image %s; skipping.", image_path)
                    continue

                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()

                keep = scores >= score_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                boxes, scores, labels = apply_nms(
                    boxes,
                    scores,
                    labels,
                    nms_iou_threshold,
                )

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    color = _color_from_label(int(label))
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    label_name = label_to_name.get(int(label), str(int(label)))
                    caption = f"{label_name} {float(score):.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(
                        caption,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        1,
                    )
                    text_x = max(0, int(x1))
                    text_y = max(text_h + baseline + 2, int(y1) - 4)
                    cv2.rectangle(
                        image,
                        (text_x, text_y - text_h - baseline),
                        (text_x + text_w, text_y + baseline),
                        color,
                        -1,
                    )
                    cv2.putText(
                        image,
                        caption,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                output_name = f"{image_id}_{image_path.name}" if image_id >= 0 else image_path.name
                output_path = output_dir / output_name
                cv2.imwrite(str(output_path), image)

    logging.info("Saved test prediction visualizations to %s", output_dir)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_metric: float,
    dataset_info: DatasetInfo,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "category_id_to_label": dataset_info.category_id_to_label,
        "label_to_name": dataset_info.label_to_name,
    }
    last_path = output_dir / "model_last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = output_dir / "model_best.pth"
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep_indices: List[torch.Tensor] = []
    for label in torch.unique(labels):
        mask = labels == label
        if mask.sum() == 0:
            continue
        label_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        label_keep = nms(boxes[mask], scores[mask], iou_threshold)
        keep_indices.append(label_indices[label_keep])

    if not keep_indices:
        return boxes[:0], scores[:0], labels[:0]

    keep = torch.cat(keep_indices)
    order = scores[keep].argsort(descending=True)
    keep = keep[order]
    return boxes[keep], scores[keep], labels[keep]


def build_dataloaders(
    args: argparse.Namespace,
    dataset_info: Optional[DatasetInfo] = None,
) -> Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader]:
    data_root = Path(args.data_root)

    def split_paths(split_name: str) -> Tuple[Path, Path]:
        split_root = data_root / split_name
        json_path = split_root / args.annotations_filename
        images_dir = split_root / args.images_dirname
        return json_path, images_dir

    train_json, train_images = split_paths(args.train_dir)
    valid_json, valid_images = split_paths(args.valid_dir)
    test_json, test_images = split_paths(args.test_dir)

    if dataset_info is None:
        train_dataset = CocoDetectionDataset(
            json_path=train_json,
            images_dir=train_images,
            transforms=get_transforms(True, args.hflip_prob),
            verify_images=True,
            skip_missing=args.skip_missing,
            limit=args.limit,
        )
        dataset_info = train_dataset.dataset_info
    else:
        train_dataset = CocoDetectionDataset(
            json_path=train_json,
            images_dir=train_images,
            transforms=get_transforms(True, args.hflip_prob),
            category_id_to_label=dataset_info.category_id_to_label,
            label_to_name=dataset_info.label_to_name,
            verify_images=True,
            skip_missing=args.skip_missing,
            limit=args.limit,
        )

    valid_dataset = CocoDetectionDataset(
        json_path=valid_json,
        images_dir=valid_images,
        transforms=get_transforms(False, 0.0),
        category_id_to_label=dataset_info.category_id_to_label,
        label_to_name=dataset_info.label_to_name,
        verify_images=True,
        skip_missing=args.skip_missing,
        limit=args.limit,
    )

    test_dataset = CocoDetectionDataset(
        json_path=test_json,
        images_dir=test_images,
        transforms=get_transforms(False, 0.0),
        category_id_to_label=dataset_info.category_id_to_label,
        label_to_name=dataset_info.label_to_name,
        verify_images=True,
        skip_missing=args.skip_missing,
        limit=args.limit,
    )

    num_workers = args.num_workers
    if os.name == "nt" and num_workers > 0:
        logging.warning("Windows detected. Consider using --num-workers 0 for stability.")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return dataset_info, train_loader, valid_loader, test_loader


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
