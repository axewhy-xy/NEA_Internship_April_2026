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

def pairwise(values: List[float]) -> List[Tuple[float, float]]:
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

def resolve_source_path(path_text: str) -> Path:
    raw_path = normalize_path(path_text)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / raw_path).resolve()

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def flip_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "horizontal":
        return ImageOps.mirror(image)
    elif mode == "vertical":
        return ImageOps.flip(image)
    else:
        raise ValueError(f"Invalid flip mode: {mode}")
    
def scale_image(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)

def translate_image(image: Image.Image, x_offset: int, y_offset: int) -> Image.Image:
    return ImageChops.offset(image, x_offset, y_offset)

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def add_noise(image: Image.Image, mean: float, stddev: float, rng: random.Random) -> Image.Image:
    if stddev < 0:
        raise ValueError("Noise stddev must be >= 0")

    mode = "L" if image.mode == "L" else "RGB"
    noisy_base = image.convert(mode)
    data = bytearray(noisy_base.tobytes())

    for idx, pixel in enumerate(data):
        noisy_value = int(round(pixel + rng.gauss(mean, stddev)))
        data[idx] = max(0, min(255, noisy_value))

    return Image.frombytes(mode, noisy_base.size, bytes(data))

def random_crop(image: Image.Image, crop_size: Tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = crop_size
    if crop_width > width or crop_height > height:
        raise ValueError("Crop size must be smaller than the original image size.")
    x_max = width - crop_width
    y_max = height - crop_height
    x_offset = rng.randint(0, x_max)
    y_offset = rng.randint(0, y_max)
    return image.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))

def random_erasing(image: Image.Image, erase_size: Tuple[int, int], erase_value: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    erase_width, erase_height = erase_size
    if erase_width > width or erase_height > height:
        raise ValueError("Erase size must be smaller than the original image size.")
    x_max = width - erase_width
    y_max = height - erase_height
    x_offset = rng.randint(0, x_max)
    y_offset = rng.randint(0, y_max)
    erased_image = image.copy()
    draw = ImageDraw.Draw(erased_image)
    draw.rectangle([x_offset, y_offset, x_offset + erase_width, y_offset + erase_height], fill=erase_value)
    return erased_image


AugmenterFn = Callable[[Image.Image, random.Random], Tuple[Image.Image, Dict[str, object]]]

def build_augmenters(args) -> List[Tuple[str, AugmenterFn]]:
    translation_pairs = [(int(x), int(y)) for x, y in pairwise(args.translation_offsets)]
    noise_pairs = pairwise(args.noise_params)

    use_specific_flags = any(
        [
            args.use_rotation,
            args.use_flip,
            args.use_scale,
            args.use_translation,
            args.use_brightness,
            args.use_contrast,
            args.use_noise,
            args.use_crop,
            args.use_erase,
        ]
    )

    use_all = args.use_all_augs or not use_specific_flags
    augmenters: List[Tuple[str, AugmenterFn]] = []

    if use_all or args.use_rotation:
        def rotate_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            angle = rng.choice(args.rotation_angles)
            return rotate_image(image, angle), {"angle": angle}
        augmenters.append(("rotation", rotate_aug))

    if use_all or args.use_flip:
        def flip_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            mode = rng.choice(args.flip_modes)
            return flip_image(image, mode), {"mode": mode}
        augmenters.append(("flip", flip_aug))

    if use_all or args.use_scale:
        def scale_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            factor = rng.choice(args.scale_factors)
            return scale_image(image, factor), {"factor": factor}
        augmenters.append(("scale", scale_aug))

    if use_all or args.use_translation:
        def translation_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            x_offset, y_offset = rng.choice(translation_pairs)
            return translate_image(image, x_offset, y_offset), {"x_offset": x_offset, "y_offset": y_offset}
        augmenters.append(("translation", translation_aug))

    if use_all or args.use_brightness:
        def brightness_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            factor = rng.choice(args.brightness_factors)
            return adjust_brightness(image, factor), {"factor": factor}
        augmenters.append(("brightness", brightness_aug))

    if use_all or args.use_contrast:
        def contrast_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            factor = rng.choice(args.contrast_factors)
            return adjust_contrast(image, factor), {"factor": factor}
        augmenters.append(("contrast", contrast_aug))

    if use_all or args.use_noise:
        def noise_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            mean, stddev = rng.choice(noise_pairs)
            return add_noise(image, mean, stddev, rng), {"mean": mean, "stddev": stddev}
        augmenters.append(("noise", noise_aug))

    if use_all or args.use_crop:
        def crop_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            return random_crop(image, tuple(args.crop_size), rng), {"crop_size": list(args.crop_size)}
        augmenters.append(("crop", crop_aug))

    if use_all or args.use_erase:
        def erase_aug(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, object]]:
            return random_erasing(image, tuple(args.erase_size), args.erase_value, rng), {
                "erase_size": list(args.erase_size),
                "erase_value": args.erase_value,
            }
        augmenters.append(("erase", erase_aug))

    return augmenters

def run():
    setup_logging()
    args = parse_arguments()

    input_metadata_path = resolve_path(args.input_dir, PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    output_image_dir = output_dir / "images"
    output_metadata_path = output_dir / args.output_metadata

    validate_args(args, input_metadata_path)
    augmenters = build_augmenters(args)
    if not augmenters:
        raise ValueError("No augmenters available. Enable at least one augmentation flag.")
