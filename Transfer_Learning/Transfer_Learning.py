"""
Transfer Learning for object detection on HDB features using satellite imagery and street view images.

Author: AxeWhy
Date:
Last Updated:

Description:
This project aims to utilize transfer learning techniques to train a model for object detection on 
HDB features using satellite imagery and street view images. The source file will contain the implementation
of the transfer learning process, providing argument inputs, retrieving the labelled dataset, and training
the pre-trained model on the labelled dataset to extract relevant features from the images.

This source file will contain both the YOLOv8 as well as the Faster R-CNN implementations for transfer learning,
allowing for a comparison of the two models' performance on the task of object detection on HDB features.

Assumptions:
- The labelled dataset is already prepared in the required format for training the models.
  (e.g., YOLO format for YOLO, COCO format for Faster R-CNN)
- The file paths for the images and the labels are correctly specified in the argument inputs.
"""

import os
import argparse
import torch
from torchvision import models, transforms