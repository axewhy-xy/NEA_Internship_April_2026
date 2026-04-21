"""
Data Augmentation Script for Extracted HDB Dataset

Author: AxeWhy
Date: 2026-04-21
Last Updated: 2026-04-21

Description:
This script performs data augmentation on the extracted HDB dataset. It applies the two data augmentation
categories. Namely, geometric transformations and photometric transformations. The script is designed to 
enhance the diversity of the dataset, which can help improve the performance of machine learning models trained 
on the dataset.

The script includes the following functions to support the data augmentation categories:
1. Geometric Transformations:
    - `rotate_image`: Rotates the image by a specified angle.
    - `flip_image`: Flips the image horizontally or vertically.
    - `scale_image`: Scales the image by a specified factor.
    - `translate_image`: Translates the image by specified x and y offsets.
2. Photometric Transformations:
    - `adjust_brightness`: Adjusts the brightness of the image by a specified factor.
    - `adjust_contrast`: Adjusts the contrast of the image by a specified factor.
    - `add_noise`: Adds random noise to the image with a specified mean and standard deviation.
    - `random_crop`: Randomly crops the image to a specified size.
    - `random_erasing`: Randomly erases a portion of the image with a specified size and value.

Usage:
python Data_Augmentation.py --input_metadata path/to/metadata.csv --output_dir path/to/output --option_flags ...
python Data_Augmentation.py --help  # For detailed usage instructions and available options
"""

import argparse
import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageOps, ImageChops, ImageDraw

# ----------------------------------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_METADATA                                      = "Dataset/Street/street_metadata_train.csv"
DEFAULT_OUTPUT_DIR                                          = "Data_Augmentation/Result"
DEFAULT_OUTPUT_METADATA                                     = "augmented_metadata.csv"
DEFAULT_SEED                                                = 2026

DEFAULT_NUM_AUGS_PER_IMAGE                                  = 3
DEFAULT_OVERWRITE                                           = False

DEFAULT_ROTATION_ANGLES                                     = [-15, -10, -5, 5, 10, 15]
DEFAULT_FLIP_MODES                                          = ["horizontal", "vertical"]
DEFAULT_SCALE_FACTORS                                       = [0.8, 0.9, 1.1, 1.2]
DEFAULT_TRANSLATION_OFFSETS                                 = [(-10, 0), (10, 0), (0, -10), (0, 10)]
DEFAULT_BRIGHTNESS_FACTORS                                  = [0.8, 0.9, 1.1, 1.2]
DEFAULT_CONTRAST_FACTORS                                    = [0.8, 0.9, 1.1, 1.2]
DEFAULT_NOISE_PARAMS                                        = [(0, 10), (0, 20), (0, 30)]
DEFAULT_CROP_SIZE                                           = (224, 224)
DEFAULT_ERASE_SIZE                                          = (32, 32)
DEFAULT_ERASE_VALUE                                         = 0

OUTPUT_COLUMNS                                              = [
                                                            "source_path",
                                                            "augmented_path",
                                                            "label",
                                                            "split",
                                                            "augmentation",
                                                            "params_json",
                                                            "seed",
                                                            "status",
                                                            "error",
                                                        ]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Augmentation Script for Extracted HDB Dataset")

    # Input/output paths and settings
    parser.add_argument("--input_dir", "--input_metadata", dest="input_dir", type=str, default=DEFAULT_INPUT_METADATA, help="Path to input metadata CSV file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save augmented images")
    parser.add_argument("--output_metadata", type=str, default=DEFAULT_OUTPUT_METADATA, help="Filename for output metadata CSV")
    parser.add_argument("--num_augs_per_image", type=int, default=DEFAULT_NUM_AUGS_PER_IMAGE, help="Number of augmentations to generate per image")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed used to deterministically generate each sample")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing augmented images and metadata")

    # Augmentation parameters
    parser.add_argument("--rotation_angles", type=int, nargs="+", default=DEFAULT_ROTATION_ANGLES, help="List of angles for rotation augmentation")
    parser.add_argument("--flip_modes", type=str, nargs="+", default=DEFAULT_FLIP_MODES, help="List of modes for flip augmentation (horizontal, vertical)")
    parser.add_argument("--scale_factors", type=float, nargs="+", default=DEFAULT_SCALE_FACTORS, help="List of factors for scale augmentation")
    parser.add_argument("--translation_offsets", type=int, nargs="+", default=[item for sublist in DEFAULT_TRANSLATION_OFFSETS for item in sublist], help="List of x and y offsets for translation augmentation (flattened)")
    parser.add_argument("--brightness_factors", type=float, nargs="+", default=DEFAULT_BRIGHTNESS_FACTORS, help="List of factors for brightness adjustment augmentation")
    parser.add_argument("--contrast_factors", type=float, nargs="+", default=DEFAULT_CONTRAST_FACTORS, help="List of factors for contrast adjustment augmentation")
    parser.add_argument("--noise_params", type=float, nargs="+", default=[item for sublist in DEFAULT_NOISE_PARAMS for item in sublist], help="List of mean and stddev pairs for noise addition augmentation (flattened)")
    parser.add_argument("--crop_size", type=int, nargs=2, default=DEFAULT_CROP_SIZE, help="Width and height for random crop augmentation")
    parser.add_argument("--erase_size", type=int, nargs=2, default=DEFAULT_ERASE_SIZE, help="Width and height for random erasing augmentation")
    parser.add_argument("--erase_value", type=int, default=DEFAULT_ERASE_VALUE, help="Pixel value to use for random erasing augmentation")

    # Option Flags
    parser.add_argument("--use_rotation", action="store_true", help="Whether to apply rotation augmentation")
    parser.add_argument("--use_flip", action="store_true", help="Whether to apply flip augmentation")
    parser.add_argument("--use_scale", action="store_true", help="Whether to apply scale augmentation")
    parser.add_argument("--use_translation", action="store_true", help="Whether to apply translation augmentation")
    parser.add_argument("--use_brightness", action="store_true", help="Whether to apply brightness adjustment augmentation")
    parser.add_argument("--use_contrast", action="store_true", help="Whether to apply contrast adjustment augmentation")
    parser.add_argument("--use_noise", action="store_true", help="Whether to apply noise addition augmentation")
    parser.add_argument("--use_crop", action="store_true", help="Whether to apply random crop augmentation")
    parser.add_argument("--use_erase", action="store_true", help="Whether to apply random erasing augmentation")
    parser.add_argument("--use_all_augs", action="store_true", help="Whether to apply all augmentation types")

    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def validate_args(args, input_metadata_path: Path):
    if args.num_augs_per_image <= 0:
        raise ValueError("num-augs-per-image must be > 0")
    if not input_metadata_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_metadata_path}")

    if len(args.translation_offsets) % 2 != 0:
        raise ValueError("translation_offsets must have an even number of values: x1 y1 x2 y2 ...")
    if len(args.noise_params) % 2 != 0:
        raise ValueError("noise_params must have an even number of values: mean1 std1 mean2 std2 ...")

    if any(scale <= 0 for scale in args.scale_factors):
        raise ValueError("All scale_factors must be > 0")
    if any(factor <= 0 for factor in args.brightness_factors):
        raise ValueError("All brightness_factors must be > 0")
    if any(factor <= 0 for factor in args.contrast_factors):
        raise ValueError("All contrast_factors must be > 0")

    if args.crop_size[0] <= 0 or args.crop_size[1] <= 0:
        raise ValueError("crop_size must contain positive width and height")
    if args.erase_size[0] <= 0 or args.erase_size[1] <= 0:
        raise ValueError("erase_size must contain positive width and height")
    if not 0 <= args.erase_value <= 255:
        raise ValueError("erase_value must be between 0 and 255")

    valid_flip_modes = {"horizontal", "vertical"}
    invalid_modes = [mode for mode in args.flip_modes if mode not in valid_flip_modes]
    if invalid_modes:
        raise ValueError(f"Invalid flip_modes: {invalid_modes}. Valid values are horizontal, vertical")

    noise_pairs = pairwise(args.noise_params)
    if any(stddev < 0 for _, stddev in noise_pairs):
        raise ValueError("All noise stddev values must be >= 0")

def normalize_path(path_text: str) -> Path:
    # Metadata currently stores Windows separators; normalize for portability.
    return Path(path_text.replace("\\", "/"))

def resolve_path(path_text: str, root: Path) -> Path:
    raw_path = normalize_path(path_text)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (root / raw_path).resolve()
