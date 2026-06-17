"""
Transfer Learning for object detection on HDB features using satellite imagery and street view images.

Author: AxeWhy
Date: 30/05/2026
Last Updated: 16/06/2026

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

To train a model with F1-based validation performance with plateau learning rate scheduling and early stopping:
python Transfer_Learning.py --run-name my_experiment --monitor-metric f1 --lr-scheduler plateau --early-stop-patience 5 --early-stop-min-delta 0.01

To resume training from a checkpoint:
python Transfer_Learning.py --run-name my_experiment --resume path/to/model_last.pth

To evaluate a trained model on the test set:
python Transfer_Learning.py --run-name my_experiment --resume path/to/model_best.pth --test-only

To run inference on unlabeled images in a directory:
python Transfer_Learning.py --run-name my_experiment --resume path/to/model_best.pth --test-deploy path/to/images --deploy-output path/to/output

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
import re
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

"""
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
Description:
This function sets up logging to both the console and a file. It creates the output directory if it 
does not exist and configures the logging format and handlers.

Args:
- output_dir (Path): The directory where the log file will be saved.

Returns:
- None

"""
def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="a", encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

"""
# ----------------------------------------------------------------------------------------------------
# set_seed()
# ----------------------------------------------------------------------------------------------------
Description:
This function sets the random seed for reproducibility across various libraries and configurations. It
ensures that the same seed is used for Python's random module, PyTorch's CPU and CUDA operations, and
configures PyTorch's cuDNN backend for deterministic behavior.

Args:
- seed (int): The random seed to set.

Returns:
- None

"""
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
# ----------------------------------------------------------------------------------------------------
# collate_fn()
# ----------------------------------------------------------------------------------------------------
Description:
This function is a custom collate function for the DataLoader. It takes a batch of data, which is a list
of tuples containing images and their corresponding targets, and separates them into two list: one for
images and one for targets. This is necessary because the default collate function does not handle batches
of dictionaries well, which is the format of the targets in this object detection task.

Args:
- batch (Iterable[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]): 
  A batch of data from the DataLoader, where each item is a tuple of an image tensor and a target dictionary.

Returns:
- Tuple[List, List]:
  A tuple containing two lists: the first list contains the image tensors, 
  and the second list contains the target dictionaries.

"""
def collate_fn(batch: Iterable[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List, List]:
    images, targets = zip(*batch)
    return list(images), list(targets)

"""
# ----------------------------------------------------------------------------------------------------
# Compose Class
# ----------------------------------------------------------------------------------------------------
Description:
This class is a custom implementation of a transform composition utility. It allows for chaining multiple
transform together. Each transform is expected to be a callable that takes an image and a target dictionary,
applies some transformation to the image and target, and returns the transformed image and target. The
Compose class iterates through the list of transforms and applies them sequentially to the input image and
target.

Args:
- transforms (List): A list of transform callables to be applied in sequence.

Returns:
- None (the class itself is callable and returns the transformed image and target when called)

"""
class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

"""
# ----------------------------------------------------------------------------------------------------
# ToTensor Class
# ----------------------------------------------------------------------------------------------------
Description:
This class is a custom transform that converts a PIL image to a PyTorch tensor. It is designed to be used
as part of a transform pipeline for object detection tasks. The __call__ method takes an image and a target
dictionary, applies the conversion to the image, and returns the transformed image along with the unchanged
target.

Args:
- None (the class itself is callable and does not require initialization parameters)

Returns:
- Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
  A tuple containing the transformed image as a PyTorch tensor and the original target dictionary.

"""
class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        return F.to_tensor(image), target

"""
# ----------------------------------------------------------------------------------------------------
# RandomHorizontalFlip Class
# ----------------------------------------------------------------------------------------------------
Description:
This class is a custom transform that randomly flips an image and its corresponding bounding boxes 
horizontally with a specified probability. The __call__ method checks if a random number is less than 
the defined probability, and if so, it applies a horizontal flip to the image and adjusts the bounding 
box coordinates accordingly. The target dictionary is updated with the new bounding box coordinates, 
and both the transformed image and target are returned.

Args:
- probability (float): The probability with which the horizontal flip will be applied. Default is 0.5.

Returns:
- Tuple[Image.Image, Dict[str, torch.Tensor]]:
  A tuple containing the potentially flipped image and the updated target dictionary with adjusted 
  bounding boxes.

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# get_transforms()
# ----------------------------------------------------------------------------------------------------
Description:
This function generates a composition of transforms to be applied to the images and targets during training
and validation. It takes a boolean indicating whether the transforms are for training and a probability for
horizontal flipping. If the transforms are for training and the horizontal flip probability is greater than
zero, it includes the RandomHorizontalFlip transform in the composition. The ToTensor transform is always
included to convert images to tensors. The function returns a Compose object that encapsulates the defined
transforms.

Args:
- train (bool): A boolean indicating whether the transforms are for training. If True, data augmentation
  may be applied.
- hflip_prob (float): The probability of applying the RandomHorizontalFlip transform. This is only
    considered if train is True.

Returns:
- Compose: A Compose object that contains the sequence of transforms to be applied to the images and targets.

"""
def get_transforms(train: bool, hflip_prob: float) -> Compose:
    transforms: List = [ToTensor()]
    if train and hflip_prob > 0:
        transforms.insert(0, RandomHorizontalFlip(hflip_prob))
    return Compose(transforms)

"""
# ----------------------------------------------------------------------------------------------------
# DatasetInfo Dataclass
# ----------------------------------------------------------------------------------------------------
Description:
This dataclass is a simple container for storing information about the dataset, specifically the mapping
from category IDs to label indices and from label indices to category names. The category_id_to_label
dictionary maps the original category IDs from the dataset to a contiguous range of label indices starting
from 1 (with 0 reserved for background). The label_to_name dictionary provides a mapping from these label
indices back to human-readable category names, which can be useful for visualization and interpretation of results.

Args:
- category_id_to_label (Dict[int, int]): A dictionary mapping original category IDs to label indices.
- label_to_name (Dict[int, str]): A dictionary mapping label indices to category names.

Returns:
- None (the dataclass itself is a simple data container and does not have behavior)

"""
@dataclass
class DatasetInfo:
    category_id_to_label: Dict[int, int]
    label_to_name: Dict[int, str]

"""
# ----------------------------------------------------------------------------------------------------
# CocoDetectionDataset Class
# ----------------------------------------------------------------------------------------------------
Description:
This class is a custom implementation of a PyTorch Dataset for loading and processing a COCO-style object
detection dataset. It reads the dataset information from a JSON file, resolves image paths, and prepares
the data for training and evaluation. The __getitem__ method loads an image, retrieves its annotations,
and applies any specified transforms to both the image and the target. The dataset is designed to be flexible
in terms of handling missing images, verifying image existence, and limiting the number of samples for
quick testing. It also constructs the target dictionary in the format expected by the Faster R-CNN model.

Args:
- json_path (Path): The path to the JSON file containing the dataset annotations.
- images_dir (Path): The directory where the images are stored.
- transforms (Optional[Compose]): A composition of transforms to be applied to the images and targets.
- category_id_to_label (Optional[Dict[int, int]]): A mapping from original category IDs to label indices.
- label_to_name (Optional[Dict[int, str]]): A mapping from label indices to category names.
- verify_images (bool): Whether to verify the existence of images when resolving paths.
- skip_missing (bool): Whether to skip records with missing images instead of raising an error.
- limit (int): An optional limit on the number of samples to load from the dataset. If 0, all samples are loaded.

Member Functions:
- __init__: Initializes the dataset by loading the JSON annotations, resolving image paths, and preparing
  the dataset information.
- _resolve_image_path: A helper function to resolve the path to an image based on the information in the
    JSON and the provided images directory.
- __len__: Returns the number of samples in the dataset.
- get_image_path: Retrieves the resolved image path for a given image ID.
- __getitem__: Loads an image and its corresponding annotations, applies transforms, and returns the image
  and target in the format expected by the model.

Returns:
- None (the class itself is a PyTorch Dataset and does not return anything upon initialization)

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# create_model()
# ----------------------------------------------------------------------------------------------------
Description:
This function creates a Faster R-CNN model with a ResNet-50 backbone and a Feature Pyramid Network (FPN). 
It allows for customization of the number of classes, whether to use pretrained weights, how many backbone
layers to train, and the minimum and maximum input sizes. The function initializes the model with the specified
parameters and replaces the default box predictor with a new one that matches the number of classes in the
dataset.

Args:
- num_classes (int): The number of classes for the object detection task, including the background class.
- pretrained (bool): Whether to use pretrained weights for the backbone and FPN.
- trainable_backbone_layers (int): The number of backbone layers to set as trainable. The rest will be frozen.
- min_size (Optional[int]): The minimum size of the input images. If None, the default value is used.
- max_size (Optional[int]): The maximum size of the input images. If None, the default value is used.

Returns:
- nn.Module: The initialized Faster R-CNN model ready for training or evaluation.

"""
def create_model(
    num_classes: int,
    pretrained: bool,
    trainable_backbone_layers: int,
    min_size: Optional[int],
    max_size: Optional[int],
) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    kwargs = {}
    if pretrained:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers
    else:
        kwargs["weights_backbone"] = None
    if min_size is not None:
        kwargs["min_size"] = min_size
    if max_size is not None:
        kwargs["max_size"] = max_size
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

"""
# ----------------------------------------------------------------------------------------------------
# move_targets_to_device()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a list of target dictionaries, where each dictionary contains tensors for bounding
boxes, labels, and other relevant information for object detection, and moves all the tensors to the
specified device (CPU or GPU). This is necessaary to ensure that the targets are on the same device as
the model during training and evaluation. The function iterates through each target dictionary and moves
each tensor to the specified device, returning a new list of target dictionaries with the tensors on the
correct device.

Args:
- targets (List[Dict[str, torch.Tensor]]): A list of target dictionaries, where each dictionary contains
  tensors for bounding boxes, labels, and other relevant information for object detection.
- device (torch.device): The device to which the tensors should be moved (e.g., torch.device('cuda') or 
  torch.device('cpu')).

Returns:
- List[Dict[str, torch.Tensor]]: A new list of target dictionaries with all tensors moved to the specified device.

"""
def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved

"""
# ----------------------------------------------------------------------------------------------------
# autocast_context()
# ----------------------------------------------------------------------------------------------------
Description:
This function provides a context manager for automatic mixed precision (AMP) based on the availability of
the torch.amp.autocast API. It checks if the autocast function is available in the current version of PyTorch
and uses it if possible. If not, it falls back to using torch.cuda.amp.autocast for CUDA devices. This allows
the code to be compatible with different versions of PyTorch while still benefiting from mixed precision training
when available. The context manager can be used to wrap the forward pass of the model during training and
evaluation to enable AMP for improved performance on supported hardware.

Args:
- device (torch.device): The device type for which to enable autocast (e.g., torch.device('cuda') or torch.device('cpu')).
- enabled (bool): Whether to enable autocast. This should typically be True during training and evaluation when AMP is desired.

Returns:
- Context manager for autocast that can be used with a "with" statement to enable mixed precision operations.

"""
def autocast_context(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")

"""
# ----------------------------------------------------------------------------------------------------
# train_one_epoch()
# ----------------------------------------------------------------------------------------------------
Description:
This function performs one epoch of training for the given model. It iterates through the data loader,
moving images and targets to the specified device, and computes the loss for each batch. The function
supports mixed precision training using a GradScaler, and applies gradient clipping if a max_norm is 
specified. It also logs the average loss at a specified frequency. The function returns the average
loss for the epoch after processing all batches.

Args:
- model (nn.Module): The model to be trained.
- optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
- data_loader (DataLoader): The DataLoader providing the training data.
- device (torch.device): The device on which to perform the training (e.g., torch.device('cuda') or
    torch.device('cpu')).
- epoch (int): The current epoch number, used for logging purposes.
- scaler (Optional[torch.cuda.amp.GradScaler]): An optional GradScaler for mixed precision training.
- max_norm (float): The maximum norm for gradient clipping. If 0, no clipping is applied.
- print_freq (int): The frequency (in steps) at which to log the average loss.

Returns:
- float: The average loss for the epoch after processing all batches.

"""
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
        with autocast_context(device, amp_enabled):
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

"""
# ----------------------------------------------------------------------------------------------------
# compute_validation_loss()
# ----------------------------------------------------------------------------------------------------
Description:
This function computes the average validation loss for the given model on the provided data loader.
It sets the model to evaluation mode, but ensures that certain modules like BatchNorm and Dropout remains
in evaluation mode to avoid issues with small batch sizes. This has to be done since loss is only evaluated
in training mode for Faster R-CNN. The function iterates through the validation data, moving images and
targets to the specified device, and computes the loss for each batch without updating the model parameters.
It also handles mixed precision evaluation if a GradScaler is enabled. The function returns the average validation
loss after processing all batches.

Args:
- model (nn.Module): The model for which to compute the validation loss.
- data_loader (DataLoader): The DataLoader providing the validation data.
- device (torch.device): The device on which to perform the evaluation (e.g., torch.device('cuda') or
    torch.device('cpu')).
- use_amp (bool): Whether to use automatic mixed precision (AMP) during evaluation.

Returns:
- float: The average validation loss after processing all batches.

"""
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
                with autocast_context(device, amp_enabled):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses):
                    raise ValueError(f"Non-finite loss detected: {losses.item():.6f}")

                running_loss += losses.item()
    finally:
        for module, was_training in module_training_states:
            module.train(was_training)

    return running_loss / max(1, len(data_loader))

"""
# ----------------------------------------------------------------------------------------------------
# evaluate()
# ----------------------------------------------------------------------------------------------------
Description:
This function evaluates the performance of the given model on the provided data loader using object
detection metrics. It computes true positives, false positives, false negatives, and mean IoU based on
the model's predictions and the ground truth annotations. The function applies a score threshold to filter
predictions, and uses non-maximum suppression (NMS) to remove duplicate detections. It then calculates 
precision, recall, F1 score, and mean IoU based on the matches between predictions and ground truth boxes.
The function returns a dictionary containing these evaluation metrics.

Args:
- model (nn.Module): The model to be evaluated.
- data_loader (DataLoader): The DataLoader providing the evaluation data.
- device (torch.device): The device on which to perform the evaluation (e.g., torch.device('cuda') or
    torch.device('cpu')).
- score_threshold (float): The minimum confidence score required for a prediction to be considered valid.
- iou_threshold (float): The IoU threshold for determining true positives when matching predictions to ground truth.
- nms_iou_threshold (float): The IoU threshold to use for non-maximum suppression to filter out duplicate detections.

Returns:
- Dict[str, float]: A dictionary containing the evaluation metrics: precision, recall, F1 score, and mean IoU.

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# resolve_monitor_mode()
# ----------------------------------------------------------------------------------------------------
Description:
This helper resolves whether lower or higher values are better for the selected validation monitor metric.
If the monitor mode is set to "auto", it assumes that "val_loss" should be minimized and all other metrics
should be maximized.

Args:
- monitor_metric (str): Validation metric being monitored.
- monitor_mode (str): Requested mode, either "auto", "min", or "max".

Returns:
- str: "min" when lower values are better, otherwise "max".
"""
def resolve_monitor_mode(monitor_metric: str, monitor_mode: str) -> str:
    if monitor_mode != "auto":
        return monitor_mode
    return "min" if monitor_metric == "val_loss" else "max"

"""
# ----------------------------------------------------------------------------------------------------
# get_monitor_value()
# ----------------------------------------------------------------------------------------------------
Description:
This helper retrieves the validation value used for checkpointing, plateau scheduling, and early stopping.
It supports both the validation loss and any of the computed validation metrics. The function checks the
selected monitor metric and returns the corresponding value from either the validation loss or the metrics
dictionary. This allows for flexible monitoring based on the user's choice of metric for evaluating model
performance during training.

Args:
- monitor_metric (str): Validation metric to retrieve.
- val_loss (float): Validation loss for the current epoch.
- metrics (Dict[str, float]): Validation detection metrics for the current epoch.

Returns:
- float: The selected validation monitor value.
"""
def get_monitor_value(
    monitor_metric: str,
    val_loss: float,
    metrics: Dict[str, float],
) -> float:
    if monitor_metric == "val_loss":
        return float(val_loss)
    return float(metrics[monitor_metric])

"""
# ----------------------------------------------------------------------------------------------------
# has_monitor_improved()
# ----------------------------------------------------------------------------------------------------
Description:
This helper compares a current validation monitor value against the previous best value. It determines
whether the monitor has improved based on the selected monitor mode ("min" for metrics where lower is
better, "max" for metrics where higher is better) and a minimum improvement threshold (min_delta). This
allows for more robust checkpointing and early stopping by ensuring that only significant improvements
are considered as true improvements, rather than minor fluctuations in the monitored metric.

Args:
- current_value (float): Current epoch monitor value.
- best_value (float): Best monitor value seen so far.
- monitor_mode (str): "min" when lower is better, "max" when higher is better.
- min_delta (float): Minimum improvement required.

Returns:
- bool: True if the monitor improved by at least min_delta.
"""
def has_monitor_improved(
    current_value: float,
    best_value: float,
    monitor_mode: str,
    min_delta: float,
) -> bool:
    if monitor_mode == "min":
        return current_value < (best_value - min_delta)
    return current_value > (best_value + min_delta)

"""
# ----------------------------------------------------------------------------------------------------
# initial_best_monitor_value()
# ----------------------------------------------------------------------------------------------------
Description:
This helper returns the starting best value for a validation monitor. For "min" mode, it returns positive
infinity so that any real monitor value will be considered an improvement. For "max" mode, it returns
negative infinity so that any real monitor value will be considered an improvement. This provides a
sensible initial value for tracking the best monitor value during training, ensuring that the first 
epoch's monitor value will always be treated as an improvement and properly initialize the best value for
subsequent comparisons.

Args:
- monitor_mode (str): "min" when lower is better, "max" when higher is better.

Returns:
- float: Positive infinity for min mode, negative infinity for max mode.
"""
def initial_best_monitor_value(monitor_mode: str) -> float:
    return float("inf") if monitor_mode == "min" else float("-inf")

"""
# ----------------------------------------------------------------------------------------------------
# best_monitor_from_checkpoint()
# ----------------------------------------------------------------------------------------------------
Description:
This helper restores the best monitored validation value from both current and older checkpoint formats.
It checks for the presence of the "best_monitor_metric" and "best_monitor_value" keys in the checkpoint
to restore the best value for the currently selected monitor metric. If those keys are not present, it
falls back to checking for older keys like "best_val_loss" for the "val_loss" metric. If the relevant
keys are not found in the checkpoint, it returns a mode-appropriate initial value using the 
initial_best_monitor_value() function. This ensures that the best monitor value is correctly initialized
when resuming training from a checkpoint, even if the checkpoint format has changed or does not contain
the expected keys for the current monitor metric.

Args:
- checkpoint (Dict): Loaded checkpoint dictionary.
- monitor_metric (str): Currently selected validation monitor metric.
- monitor_mode (str): Currently selected monitor mode.

Returns:
- float: Restored best monitor value, or a mode-appropriate initial value if unavailable.
"""
def best_monitor_from_checkpoint(
    checkpoint: Dict,
    monitor_metric: str,
    monitor_mode: str,
) -> float:
    checkpoint_metric = checkpoint.get("best_monitor_metric")
    if checkpoint_metric == monitor_metric and "best_monitor_value" in checkpoint:
        return float(checkpoint["best_monitor_value"])

    if monitor_metric == "val_loss" and "best_val_loss" in checkpoint:
        return float(checkpoint["best_val_loss"])
    if monitor_metric == "f1":
        return float(checkpoint.get("best_f1", checkpoint.get("best_metric", 0.0)))

    return initial_best_monitor_value(monitor_mode)

TRAINING_HISTORY_KEYS = ("train_loss", "val_loss", "precision", "recall", "f1", "mean_iou")

"""
# ----------------------------------------------------------------------------------------------------
# create_training_history()
# ----------------------------------------------------------------------------------------------------
Description:
This helper creates the in-memory structure used for training plots. The explicit epoch axis allows
plots to continue with the true epoch numbers after resuming from a checkpoint.

Args:
- None

Returns:
- Dict[str, List[float]]: Empty plotting history with epoch and metric lists.
"""
def create_training_history() -> Dict[str, List[float]]:
    history = {"epochs": []}
    for key in TRAINING_HISTORY_KEYS:
        history[key] = []
    return history

def copy_training_history(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {key: list(values) for key, values in history.items()}

def normalize_training_history(
    raw_history,
    max_epoch: Optional[int] = None,
) -> Dict[str, List[float]]:
    history = create_training_history()
    if not isinstance(raw_history, dict):
        return history

    for key in ("epochs", *TRAINING_HISTORY_KEYS):
        values = raw_history.get(key, [])
        if not isinstance(values, list):
            continue

        normalized_values = []
        for value in values:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            normalized_values.append(int(numeric_value) if key == "epochs" else numeric_value)
        history[key] = normalized_values

    value_count = max((len(history[key]) for key in TRAINING_HISTORY_KEYS), default=0)
    if value_count and len(history["epochs"]) != value_count:
        history["epochs"] = list(range(1, value_count + 1))

    if max_epoch is not None and history["epochs"]:
        keep_indices = [
            index for index, epoch in enumerate(history["epochs"])
            if int(epoch) <= max_epoch
        ]
        history["epochs"] = [history["epochs"][index] for index in keep_indices]
        for key in TRAINING_HISTORY_KEYS:
            values = history.get(key, [])
            history[key] = [values[index] for index in keep_indices if index < len(values)]

    return history

def append_training_history(
    history: Dict[str, List[float]],
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict[str, float],
) -> None:
    epoch = int(epoch)
    history.setdefault("epochs", [])
    for key in TRAINING_HISTORY_KEYS:
        history.setdefault(key, [])

    cutoff = len(history["epochs"])
    for index, existing_epoch in enumerate(history["epochs"]):
        if int(existing_epoch) >= epoch:
            cutoff = index
            break

    if cutoff < len(history["epochs"]):
        history["epochs"] = history["epochs"][:cutoff]
        for key in TRAINING_HISTORY_KEYS:
            history[key] = history[key][:cutoff]

    history["epochs"].append(epoch)
    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["precision"].append(float(metrics["precision"]))
    history["recall"].append(float(metrics["recall"]))
    history["f1"].append(float(metrics["f1"]))
    history["mean_iou"].append(float(metrics["mean_iou"]))

def load_training_history_from_log(
    log_path: Path,
    max_epoch: Optional[int] = None,
) -> Dict[str, List[float]]:
    history = create_training_history()
    if not log_path.exists():
        return history

    metric_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
        r"Train loss\s+(?P<train_loss>[-+0-9.eE]+)\s+\|\s+"
        r"Val loss\s+(?P<val_loss>[-+0-9.eE]+)\s+\|\s+"
        r"Val precision\s+(?P<precision>[-+0-9.eE]+)\s+\|\s+"
        r"Val recall\s+(?P<recall>[-+0-9.eE]+)\s+\|\s+"
        r"Val F1\s+(?P<f1>[-+0-9.eE]+)\s+\|\s+"
        r"Val mIoU\s+(?P<mean_iou>[-+0-9.eE]+)"
    )
    metrics_by_epoch: Dict[int, Dict[str, float]] = {}
    try:
        with log_path.open("r", encoding="utf-8") as file:
            for line in file:
                match = metric_pattern.search(line)
                if match is None:
                    continue
                epoch = int(match.group("epoch"))
                if max_epoch is not None and epoch > max_epoch:
                    continue
                metrics_by_epoch[epoch] = {
                    key: float(match.group(key))
                    for key in TRAINING_HISTORY_KEYS
                }
    except OSError as exc:
        logging.warning("Could not read plotting history from %s: %s", log_path, exc)
        return history

    for epoch in sorted(metrics_by_epoch):
        history["epochs"].append(epoch)
        for key in TRAINING_HISTORY_KEYS:
            history[key].append(metrics_by_epoch[epoch][key])

    return history

def restore_training_history(
    checkpoint: Dict,
    resume_path: Path,
    output_dir: Path,
    max_epoch: int,
) -> Tuple[Dict[str, List[float]], str]:
    raw_history = checkpoint.get("training_history", checkpoint.get("history"))
    history = normalize_training_history(raw_history, max_epoch=max_epoch)
    if history["train_loss"]:
        return history, "checkpoint"

    seen_log_paths = set()
    for log_path in (resume_path.parent / "train.log", output_dir / "train.log"):
        resolved_log_path = str(log_path.resolve())
        if resolved_log_path in seen_log_paths:
            continue
        seen_log_paths.add(resolved_log_path)

        history = load_training_history_from_log(log_path, max_epoch=max_epoch)
        if history["train_loss"]:
            return history, str(log_path)

    return create_training_history(), ""

"""
# ----------------------------------------------------------------------------------------------------
# get_plotting_backend()
# ----------------------------------------------------------------------------------------------------
Description:
This function attempts to import the Matplotlib plotting library and returns the pyplot module if the
import is successful. If Matplotlib is not installed, the function returns None. This allows the rest of
the code to check for the availability of Matplotlib before attempting to generate plots, and to handle
the absence of the library gracefully without crashing.

Args:
- None

Returns:
- Optional[ModuleType]: The Matplotlib pyplot module if available, or None if Matplotlib is not installed.
"""
def get_plotting_backend():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt

"""
# ----------------------------------------------------------------------------------------------------
# save_training_plots()
# ----------------------------------------------------------------------------------------------------
Description:
This function generates and saves training and validation loss curves, as well as validation metrics
curves, based on the provided training history. It takes a history dictionary containing list of epoch,
loss, and metric values, an output directory to save the plots, and the Matplotlib pyplot module.
The function checks if the necessary data is available in the history and generates the corresponding plots.
The loss curve plot shows the training and validation loss over epochs, while the metrics curve plot shows
the precision, recall, F1 score, and mean IoU over epochs. The generated plots are saved as PNG files in the
specified output directory. If the history does not contain the required data for plotting, the function 
will simply return without generating any plots.

Args:
- history (Dict[str, List[float]]): A dictionary containing lists of epoch, loss, and metric values.
  Expected keys include "epochs", "train_loss", "val_loss", "precision", "recall", "f1", and "mean_iou".
- output_dir (Path): The directory where the generated plots should be saved.
- plt: The Matplotlib pyplot module used for plotting. This should be obtained from the get_plotting_backend() function.

Returns:
- None (the function generates and saves plots but does not return any value)

"""
def save_training_plots(
    history: Dict[str, List[float]],
    output_dir: Path,
    plt,
) -> None:
    if not history.get("train_loss"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epochs", [])
    if len(epochs) != len(history["train_loss"]):
        epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    if history.get("val_loss"):
        ax.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="Val loss")
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

"""
# ----------------------------------------------------------------------------------------------------
# _color_from_label()
# ----------------------------------------------------------------------------------------------------
Description:
This helper function generates a consistent RGB color tuple based on an integer label. It uses simple
modulo operations with prime numbers to create a pseudo-random but deterministic color for each label.
This can be useful for visualizing predictions with different colors for different classes, ensuring
that the same label always gets the same color across different images and runs.

Args:
- label (int): The integer label for which to generate a color.

Returns:
- Tuple[int, int, int]: A tuple representing the RGB color corresponding to the given label.

"""
def _color_from_label(label: int) -> Tuple[int, int, int]:
    return (
        int((37 * label) % 255),
        int((17 * label) % 255),
        int((29 * label) % 255),
    )

"""
# ----------------------------------------------------------------------------------------------------
# _is_relative_to()
# ----------------------------------------------------------------------------------------------------
Description:
This helper function checks if a given path is relative to a specified parent path. It attempts to compute
the relative path from the given path to the parent path using the Path.relative_to() method. If the given
path is indeed relative to the parent, the method will succeed and the function returns True. If the given
path is not relative to the parent, a ValueError will be raised, which is caught by the function, and it 
returns False. This function can be used to verify that resolved image paths are within expected directories.

Args:
- path (Path): The path to check for being relative to the parent.
- parent (Path): The parent path to which the given path should be relative.

Returns:
- bool: True if the given path is relative to the parent path, False otherwise.

"""
def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True

"""
# ----------------------------------------------------------------------------------------------------
# visualize_predictions()
# ----------------------------------------------------------------------------------------------------
Description:
This function visualizes the predictions of the model on a given dataset and saves the resulting images
with bounding boxes drawn around the detected objects. It takes the model, a data loader for the dataset,
the dataset object itself, the device to run the model on, an output directory to save the visualizations,
the score threshold for filtering predictions, IoU threshold for non-maximum suppression, and a mapping
from label indices to category names. The function iterates through the data loader, obtains the model's 
predictions, applies the score threshold and NMS to filter the predictions, and then draws bounding boxes
and labels on the original images using OpenCV. The visualized images are saved to the specified output
directory. If OpenCV is not installed, the function raises a RuntimeError indicating that OpenCV is required for
prediction visualization.

Args:
- model (nn.Module): The trained model to use for generating predictions.
- data_loader (DataLoader): The DataLoader providing the dataset to visualize.
- dataset (CocoDetectionDataset): The dataset object, used to resolve image paths for visualization.
- device (torch.device): The device on which to run the model for generating predictions (e.g., torch.device('cuda') or
    torch.device('cpu')).
- output_dir (Path): The directory where the visualized images should be saved.
- score_threshold (float): The minimum confidence score required for a prediction to be visualized.
- nms_iou_threshold (float): The IoU threshold to use for non-maximum suppression when filtering predictions.
- label_to_name (Dict[int, str]): A mapping from label indices to human-readable category names for labeling the visualizations.

Returns:
- None (the function generates and saves visualized images but does not return any value)

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# run_deploy_inference()
# ----------------------------------------------------------------------------------------------------
Description:
This function runs inference on a directory of unlabeled images and saves annotated predictions.
It loads images from the given directory (recursively), runs the model to get detections, applies
score thresholding and NMS, and draws bounding boxes and labels onto the images using OpenCV.

Args:
- model (nn.Module): The trained model to use for inference.
- images_dir (Path): Root directory containing images to run inference on.
- output_dir (Path): Directory where annotated images will be saved.
- device (torch.device): Device to run the model on.
- score_threshold (float): Minimum confidence score for predictions.
- nms_iou_threshold (float): IoU threshold for non-maximum suppression.
- label_to_name (Dict[int, str]): Mapping from label indices to names.
- batch_size (int): Batch size for inference.
- use_amp (bool): Whether to use automatic mixed precision for inference.

Returns:
- None

"""
def run_deploy_inference(
    model: nn.Module,
    images_dir: Path,
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
    nms_iou_threshold: float,
    label_to_name: Dict[int, str],
    batch_size: int,
    use_amp: bool,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for deployment visualization.") from exc

    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == images_dir or _is_relative_to(output_dir, images_dir):
        raise ValueError(
            "Deployment output directory must be outside the input image directory "
            "to avoid overwriting or re-processing input images."
        )

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in valid_exts
        ]
    )

    if not image_paths:
        logging.warning("No images found in %s", images_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    def iter_batches(items: List[Path], size: int):
        step = max(1, size)
        for i in range(0, len(items), step):
            yield items[i : i + step]

    with torch.no_grad():
        for batch_paths in iter_batches(image_paths, batch_size):
            batch_items: List[Tuple[Path, torch.Tensor]] = []
            for image_path in batch_paths:
                try:
                    pil_image = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    logging.warning("Failed to open image %s: %s", image_path, exc)
                    continue
                batch_items.append((image_path, F.to_tensor(pil_image)))

            if not batch_items:
                continue

            images = [item[1].to(device) for item in batch_items]
            with autocast_context(device, amp_enabled):
                outputs = model(images)

            for (image_path, _), output in zip(batch_items, outputs):
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

                try:
                    rel_path = image_path.relative_to(images_dir)
                    output_path = output_dir / rel_path
                except ValueError:
                    output_path = output_dir / image_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output_path), image):
                    logging.warning("Failed to write deploy output %s", output_path)

    logging.info("Saved deploy predictions to %s", output_dir)

"""
# ----------------------------------------------------------------------------------------------------
# save_checkpoint()
# ----------------------------------------------------------------------------------------------------
Description:
This function saves the training checkpoint for the model, including the model state, optimizer state,
scheduler state, current epoch, best monitored validation value, best validation loss, best F1 score,
early-stopping state, plotting history, and dataset information. The checkpoint is saved to the specified
output directory, with a filename indicating whether it is the last checkpoint or the best checkpoint based
on the selected monitor metric. The function creates the output directory if it does not exist, and saves
the checkpoint using PyTorch's torch.save() function. The checkpoint includes all necessary information to
resume training or to evaluate the model later.

Args:
- output_dir (Path): The directory where the checkpoint should be saved.
- model (nn.Module): The model whose state is to be saved.
- optimizer (torch.optim.Optimizer): The optimizer whose state is to be saved.
- scheduler (Optional): The learning rate scheduler whose state
    is to be saved, if applicable.
- epoch (int): The current epoch number to be saved in the checkpoint.
- best_monitor_value (float): The best value achieved so far for the selected validation monitor metric.
- monitor_metric (str): The validation metric used for best-checkpoint and early-stopping decisions.
- monitor_mode (str): Whether lower ("min") or higher ("max") monitor values are better.
- best_val_loss (float): The best validation loss achieved so far.
- best_f1 (float): The best F1 score achieved so far.
- epochs_without_improve (int): Number of consecutive epochs without enough monitor improvement.
- dataset_info (DatasetInfo): The dataset information containing category and label mappings to
     be saved in the checkpoint.
- history (Dict[str, List[float]]): Training history used to resume graphical plots.
- is_best (bool): A flag indicating whether this checkpoint is the best one based on the metric,
     which determines the filename.

Returns:
- None (the function saves the checkpoint to disk but does not return any value)

"""
def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_monitor_value: float,
    monitor_metric: str,
    monitor_mode: str,
    best_val_loss: float,
    best_f1: float,
    epochs_without_improve: int,
    dataset_info: DatasetInfo,
    history: Dict[str, List[float]],
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_monitor_value,
        "best_monitor_value": best_monitor_value,
        "best_monitor_metric": monitor_metric,
        "best_monitor_mode": monitor_mode,
        "best_f1": best_f1,
        "best_val_loss": best_val_loss,
        "epochs_without_improve": epochs_without_improve,
        "category_id_to_label": dataset_info.category_id_to_label,
        "label_to_name": dataset_info.label_to_name,
        "training_history": copy_training_history(history),
    }
    last_path = output_dir / "model_last.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = output_dir / "model_best.pth"
        torch.save(state, best_path)

"""
# ----------------------------------------------------------------------------------------------------
# load_checkpoint()
# ----------------------------------------------------------------------------------------------------
Description:
This function loads a training checkpoint for the model, including the model state, optimizer state,
scheduler state, and dataset information. The checkpoint is loaded from the specified path, and the model,
optimizer, and scheduler are updated with the loaded states. The function returns the loaded checkpoint
as a dictionary, which includes the epoch number, best metric value, and dataset information. This allows
for resuming the training process or evaluating the model later with the same dataset information.

Args:
- checkpoint_path (Path): The path to the checkpoint file to be loaded.
- model (nn.Module): The model into which the state should be loaded.
- optimizer (Optional[torch.optim.Optimizer]): The optimizer into which the state should be loaded,
     if applicable.
- scheduler (Optional): The learning rate scheduler into which
     the state should be loaded, if applicable.

Returns:
- Dict: A dictionary containing the loaded checkpoint information, including epoch number, best metric
     value, and dataset information.
"""
def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except (KeyError, TypeError, ValueError) as exc:
            logging.warning("Skipping scheduler state from checkpoint because it is incompatible: %s", exc)
    return checkpoint

"""
# ----------------------------------------------------------------------------------------------------
# _infer_num_classes_from_state_dict()
# ----------------------------------------------------------------------------------------------------
Description:
This helper function attempts to infer the number of classes from a Faster R-CNN state dict by
inspecting the classifier layer dimensions. It checks for the presence of the
"roi_heads.box_predictor.cls_score.weight" and "roi_heads.box_predictor.cls_score.bias" keys in 
the state dictionary, which correspond to the weights and biases of the classification layer in the ROI
heads of the Faster R-CNN model. If either of these keys is found and has a valid shape, the function
returns the number of classes inferred from the shape of the weights or biases. If neither key is found
or if they do not have valid shapes, the function returns None, indicating that the number of classes
could not be inferred from the state dictionary.

Args:
- state_dict (Dict[str, torch.Tensor]): Model state dict.

Returns:
- Optional[int]: The inferred number of classes if available.

"""
def _infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    key = "roi_heads.box_predictor.cls_score.weight"
    if key in state_dict and hasattr(state_dict[key], "shape"):
        return int(state_dict[key].shape[0])
    key = "roi_heads.box_predictor.cls_score.bias"
    if key in state_dict and hasattr(state_dict[key], "shape"):
        return int(state_dict[key].shape[0])
    return None

"""
# ----------------------------------------------------------------------------------------------------
# apply_nms()
# ----------------------------------------------------------------------------------------------------
Description:
This function applies non-maximum suppression (NMS) to the predicted bounding boxes, scores, and labels.
It processes the predictions for each unique label separately, applies NMS to filter out duplicate detections,
and then combines the results. The function returns the filtered boxes, scores, and labels after NMS. If
there are no boxes to process, it returns empty tensors.

Args:
- boxes (torch.Tensor): A tensor of shape (N, 4) containing the predicted bounding boxes in (x1, y1, x2, y2) format.
- scores (torch.Tensor): A tensor of shape (N,) containing the confidence scores for each predicted box.
- labels (torch.Tensor): A tensor of shape (N,) containing the predicted class labels for each box.
- iou_threshold (float): The IoU threshold to use for non-maximum suppression. Boxes with IoU greater than
    this threshold will be suppressed.

Returns:
- Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the filtered boxes, scores, and labels
    after applying non-maximum suppression.

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# build_dataloaders()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds the training, validation, and test DataLoaders for the COCO-style dataset based on
the provided arguments. It reads the JSON annotations and image directories for each split, creates the
corresponding CocoDetectionDataset instances, and then constructs DataLoaders with the specified batch size,
number of workers, and other settings. The function also handles the case where a DatasetInfo object is 
provided, ensuring that the same category and label mappings are used across all splits. It returns the 
dataset information along with the three DataLoaders for training, validation, and testing.

Args:
- args (argparse.Namespace): The command-line arguments containing dataset paths, batch size, 
    number of workers, and other settings.
- dataset_info (Optional[DatasetInfo]): An optional DatasetInfo object containing category 
    and label mappings.

Returns:
- Tuple[DatasetInfo, DataLoader, DataLoader, DataLoader]: A tuple containing the dataset information
 and the DataLoaders for training, validation, and testing.

"""
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

"""
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
Description:
This function parses the command-line arguments for the training and evaluation script. It sets up various
arguments related to dataset paths, training hyperparameters, model configuration, evaluation settings, and
other options. The function also provides default values for most of the arguments, and constructs default
paths for the dataset and output directories based on the script's location. The parsed arguments are returned
as an argparse.Namespace object, which can be used throughout the script to access the configuration settings
for training and evaluation.

Args:
- None (the function reads command-line arguments directly)

Returns:
- argparse.Namespace: An object containing the parsed command-line arguments for configuring the
 training and evaluation process.

"""
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
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay for the optimizer.")
    parser.add_argument("--lr-step-size", type=int, default=3, help="Step size for the learning rate scheduler.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Gamma for the learning rate scheduler.")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="step",
        choices=["step", "plateau", "cosine", "none"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument("--plateau-patience", type=int, default=3, help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau.")
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6, help="Minimum LR for ReduceLROnPlateau.")
    parser.add_argument("--cosine-t-max", type=int, default=0, help="T_max for CosineAnnealingLR (0 uses epochs).")
    parser.add_argument("--cosine-min-lr", type=float, default=0.0, help="Minimum LR for CosineAnnealingLR.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "precision", "recall", "f1", "mean_iou"],
        help="Validation metric used for best checkpoint, ReduceLROnPlateau, and early stopping.",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        default="auto",
        choices=["auto", "min", "max"],
        help="Whether lower or higher monitor values are better (auto uses min for val_loss, max otherwise).",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer to use for training."
    )

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
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop training if the monitored validation metric does not improve for this many epochs (0 disables).")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum monitored-metric improvement required to reset early stopping patience.")

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from.")
    parser.add_argument("--test-only", action="store_true", help="Run evaluation on the test dataset only.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test evaluation after training.")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs.")
    parser.add_argument("--test-deploy", type=str, default="", help="Directory of unlabeled images for deployment inference. If set, training and test evaluation are skipped.")
    parser.add_argument("--deploy-output", type=str, default="", help="Output directory for deployment predictions (default: <output_root>/<run_name>/deploy_predictions).")

    args = parser.parse_args()
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        parser.error("--early-stop-min-delta must be >= 0")
    if args.lr_step_size <= 0:
        parser.error("--lr-step-size must be > 0")
    if args.plateau_factor <= 0 or args.plateau_factor >= 1:
        parser.error("--plateau-factor must be greater than 0 and less than 1")

    return args

"""
# ----------------------------------------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------------------------------------
Description:
This is the main function that orchestrates the entire training and evaluation process. The function
starts by parsing command-line arguments, setting up logging, and determining the device to use for
training. It then builds the DataLoaders for training, validation, and testing using the provided dataset
paths and configurations. The model is created based on the specified number of classes and whether to
use pre-trained weights. The optimizer and learning rate scheduler are set up according to the provided 
hyperparameters. The function handles resuming from a checkpoint if specified, and then enters the training
loop for the specified number of epochs. During each epoch, it trains the model, compute validation loss,
evaluates the model on the validation set, and saves checkpoints based on the performance. After training
is complete, it evaluates the best model on the test set and visualizes the predictions.

Args:
- None (the function reads command-line arguments directly)

Returns:
- None (the function performs training and evaluation but does not return any value)

"""
def main() -> None:
    args = parse_args()
    run_id = args.run_name.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_root) / run_id
    setup_logging(output_dir)

    if args.optimizer in ["adam", "adamw"] and args.learning_rate > 0.001:
        logging.warning("Selected optimizer %s typically uses a lower learning rate than %.6g; consider reducing --learning-rate for better performance.", args.optimizer, args.learning_rate)

    logging.info("Starting run %s", run_id)
    logging.info("Arguments: %s", args)

    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        logging.info("Using device: %s", device)

    if args.test_deploy:
        deploy_dir = Path(args.test_deploy)
        if not deploy_dir.exists() or not deploy_dir.is_dir():
            logging.error("Deployment directory does not exist or is not a directory: %s", deploy_dir)
            return

        checkpoint_path: Optional[Path] = None
        if args.resume:
            checkpoint_path = Path(args.resume)
        elif args.run_name:
            run_dir = Path(args.output_root) / args.run_name
            best_candidate = run_dir / "model_best.pth"
            last_candidate = run_dir / "model_last.pth"
            if best_candidate.exists():
                checkpoint_path = best_candidate
            elif last_candidate.exists():
                logging.warning("Best checkpoint not found; using %s", last_candidate)
                checkpoint_path = last_candidate

        if checkpoint_path is None or not checkpoint_path.exists():
            logging.error("Checkpoint not found. Provide --resume or ensure <output_root>/<run_name>/model_best.pth exists.")
            return

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_data.get("model_state", checkpoint_data)
        raw_label_to_name = checkpoint_data.get("label_to_name", {}) or {}
        label_to_name: Dict[int, str] = {}
        if isinstance(raw_label_to_name, dict):
            for key, value in raw_label_to_name.items():
                try:
                    label_id = int(key)
                except (TypeError, ValueError):
                    continue
                label_to_name[label_id] = str(value)

        num_classes = len(label_to_name) + 1 if label_to_name else None
        if num_classes is None:
            raw_category_id_to_label = checkpoint_data.get("category_id_to_label", {}) or {}
            fallback_label_to_name: Dict[int, str] = {}
            if isinstance(raw_category_id_to_label, dict):
                for cat_id, label in raw_category_id_to_label.items():
                    try:
                        label_id = int(label)
                    except (TypeError, ValueError):
                        continue
                    fallback_label_to_name[label_id] = str(cat_id)
            if fallback_label_to_name:
                label_to_name = fallback_label_to_name
                num_classes = len(label_to_name) + 1
            else:
                num_classes = _infer_num_classes_from_state_dict(state_dict)

        if num_classes is None:
            logging.error("Checkpoint missing label mappings and model metadata; cannot infer num_classes.")
            return
        if not label_to_name:
            logging.warning("Label names not found in checkpoint; using numeric labels.")

        model = create_model(
            num_classes=num_classes,
            pretrained=False,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_size=args.min_size,
            max_size=args.max_size,
        )
        model.load_state_dict(state_dict)
        model.to(device)

        deploy_output = Path(args.deploy_output) if args.deploy_output else output_dir / "deploy_predictions"
        try:
            run_deploy_inference(
                model=model,
                images_dir=deploy_dir,
                output_dir=deploy_output,
                device=device,
                score_threshold=args.score_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                label_to_name=label_to_name,
                batch_size=args.batch_size,
                use_amp=args.amp,
            )
        except (RuntimeError, ValueError) as exc:
            logging.error("%s", exc)
        return

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

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    monitor_mode = resolve_monitor_mode(args.monitor_metric, args.monitor_mode)
    logging.info(
        "Monitoring validation %s for best checkpoint, plateau scheduling, and early stopping (%s is better).",
        args.monitor_metric,
        "lower" if monitor_mode == "min" else "higher",
    )

    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )
    elif args.lr_scheduler == "cosine":
        t_max = args.cosine_t_max if args.cosine_t_max > 0 else args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=args.cosine_min_lr,
        )
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_f1 = 0.0
    best_val_loss = float("inf")
    best_monitor_value = initial_best_monitor_value(monitor_mode)
    epochs_without_improve = 0
    start_epoch = 1
    history = create_training_history()

    if args.resume:
        if args.test_only:
            load_checkpoint(Path(args.resume), model)
            logging.info("Loaded model weights from %s", args.resume)
        else:
            logging.warning("--resume flag is set without --test-only; Please ensure the optimizer param group keys match the checkpoint (e.g., same --trainable-backbone-layers, --min-size, --max-size, and pretrained settings as the original run).")
            resume_path = Path(args.resume)
            checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_f1 = checkpoint.get("best_f1", checkpoint.get("best_metric", 0.0))
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            best_monitor_value = best_monitor_from_checkpoint(
                checkpoint,
                args.monitor_metric,
                monitor_mode,
            )
            epochs_without_improve = int(checkpoint.get("epochs_without_improve", 0))
            history, history_source = restore_training_history(
                checkpoint=checkpoint,
                resume_path=resume_path,
                output_dir=output_dir,
                max_epoch=start_epoch - 1,
            )
            if history_source:
                logging.info(
                    "Restored plotting history for %d epoch(s) from %s.",
                    len(history["train_loss"]),
                    history_source,
                )
            elif start_epoch > 1:
                logging.warning(
                    "No plotting history found in checkpoint or train.log; graphical plots will include only newly trained epochs."
                )
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

    plot_backend = get_plotting_backend()
    graphical_dir = output_dir / "graphical_results"
    if plot_backend is None:
        logging.warning("matplotlib not available; skipping graphical results.")
    elif history["train_loss"]:
        save_training_plots(history, graphical_dir, plot_backend)


    for epoch in range(start_epoch, args.epochs + 1):
        start_time = datetime.now()
        logging.info("Training started for %d/%d epochs...", epoch, args.epochs)

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

        logging.info("Completed epoch %d training in %.2f seconds. Computing validation metrics...", epoch, (datetime.now() - start_time).total_seconds())

        val_loss = compute_validation_loss(
            model=model,
            data_loader=valid_loader,
            device=device,
            use_amp=args.amp,
        )

        logging.info("Validation loss for epoch %d: %.4f. Evaluating metrics...", epoch, val_loss)

        metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )
        monitor_value = get_monitor_value(args.monitor_metric, val_loss, metrics)

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        logging.info(
            "Epoch %d | Train loss %.4f | Val loss %.4f | Val precision %.4f | Val recall %.4f | Val F1 %.4f | Val mIoU %.4f | Monitor %s %.4f | LR %s",
            epoch,
            train_loss,
            val_loss,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["mean_iou"],
            args.monitor_metric,
            monitor_value,
            ", ".join(f"{group['lr']:.6g}" for group in optimizer.param_groups),
        )

        append_training_history(history, epoch, train_loss, val_loss, metrics)

        if plot_backend is not None:
            save_training_plots(history, graphical_dir, plot_backend)

        is_best = has_monitor_improved(monitor_value, best_monitor_value, monitor_mode, 0.0)
        resets_patience = has_monitor_improved(
            monitor_value,
            best_monitor_value,
            monitor_mode,
            args.early_stop_min_delta,
        )
        if is_best:
            best_monitor_value = monitor_value
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if resets_patience:
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_monitor_value=best_monitor_value,
            monitor_metric=args.monitor_metric,
            monitor_mode=monitor_mode,
            best_val_loss=best_val_loss,
            best_f1=best_f1,
            epochs_without_improve=epochs_without_improve,
            dataset_info=dataset_info,
            history=history,
            is_best=is_best,
        )

        if args.save_every and epoch % args.save_every == 0:
            extra_path = output_dir / f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), extra_path)

        if args.early_stop_patience > 0 and epochs_without_improve >= args.early_stop_patience:
            logging.info(
                "Early stopping triggered at epoch %d (no %s improvement >= %.6g for %d epochs).",
                epoch,
                args.monitor_metric,
                args.early_stop_min_delta,
                epochs_without_improve,
            )
            break

        logging.info("Epoch %d completed. Time taken: %.2f seconds.", epoch, (datetime.now() - start_time).total_seconds())

    logging.info(
        "Training complete. Best %s: %.4f | Best F1: %.4f | Best val loss: %.4f",
        args.monitor_metric,
        best_monitor_value,
        best_f1,
        best_val_loss,
    )

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

# Main Entry Point
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
