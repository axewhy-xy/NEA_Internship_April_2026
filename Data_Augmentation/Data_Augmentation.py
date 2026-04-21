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
'''
# ----------------------------------------------------------------------------------------------------
# parse_arguments()
# ----------------------------------------------------------------------------------------------------
Description:
This function defines and parses command-line arguments for the data augmentation script. It uses the
argparse library to specify various options for input/output paths, augmentation parameters, and flags
to enable or disable specific augmentation types. The parsed arguments are returned as a Namespace object.

Arguments:
- --input_dir / --input_metadata: Path to the input metadata CSV file (default:
    "Dataset/Street/street_metadata_train.csv").
- --output_dir: Directory to save augmented images (default: "Data_Augmentation/Result").
- --output_metadata: Filename for output metadata CSV (default: "augmented_metadata.csv").
- --num_augs_per_image: Number of augmentations to generate per image (default: 3).
- --seed: Global random seed used to deterministically generate each sample (default: 2026).
- --overwrite: Whether to overwrite existing augmented images and metadata (default: False).
- --rotation_angles: List of angles for rotation augmentation (default: [-15, -10, -5, 5, 10, 15]).
- --flip_modes: List of modes for flip augmentation (horizontal, vertical) (default:
    ["horizontal", "vertical"]).
- --scale_factors: List of factors for scale augmentation (default: [0.8, 0.9, 1.1, 1.2]).
- --translation_offsets: List of x and y offsets for translation augmentation (flattened) (default:
    [(-10, 0), (10, 0), (0, -10), (0, 10)]).
- --brightness_factors: List of factors for brightness adjustment augmentation (default:
    [0.8, 0.9, 1.1, 1.2]).
- --contrast_factors: List of factors for contrast adjustment augmentation (default:
    [0.8, 0.9, 1.1, 1.2]).
- --noise_params: List of mean and stddev pairs for noise addition augmentation (flattened) (default:
    [(0, 10), (0, 20), (0, 30)]).
- --crop_size: Width and height for random crop augmentation (default: (224, 224)).
- --erase_size: Width and height for random erasing augmentation (default: (32, 32)).
- --erase_value: Pixel value to use for random erasing augmentation (default: 0).
- --use_rotation: Whether to apply rotation augmentation (default: False).
- --use_flip: Whether to apply flip augmentation (default: False).
- --use_scale: Whether to apply scale augmentation (default: False).
- --use_translation: Whether to apply translation augmentation (default: False).
- --use_brightness: Whether to apply brightness adjustment augmentation (default: False).
- --use_contrast: Whether to apply contrast adjustment augmentation (default: False).
- --use_noise: Whether to apply noise addition augmentation (default: False).
- --use_crop: Whether to apply random crop augmentation (default: False).
- --use_erase: Whether to apply random erasing augmentation (default: False).
- --use_all_augs: Whether to apply all augmentation types (default: False). If no specific flags are
    enabled, all augmentations will be applied by default.

Returns:
- argparse.Namespace: Parsed command-line arguments.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
Description:
This function sets up the logging configuration for the script. It configures the logging level to INFO
and specifies the log message format to include the timestamp, log level, and message content. The logging
configuration allows the script to output informative messages about the progress and status of the data
augmentation process.

Arguments:
- None

Returns:
- None
'''
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

'''
# ----------------------------------------------------------------------------------------------------
# validate_args()
# ----------------------------------------------------------------------------------------------------
Description:
This function validates the command-line arguments provided to the script. It checks for conditions such
as positive values for the number of augmentations per image, existence of the input metadata file, correct
formatting of translation offsets and noise parameters, positive values for scale, brightness, and contrast
factors, valid crop and erase sizes, valid erase values, and valid flip modes. If any validation check fails,
the function raises an appropriate exception with a descriptive error message.

Arguments:
- args: The parsed command-line arguments to validate.
- input_metadata_path: The resolved Path object for the input metadata CSV file.

Returns:
- None

Raises:
- ValueError: If any of the validation checks fail, such as non-positive values, 
              incorrect formatting, or invalid options.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# normalize_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a path string as input and normalizes it by replacing Windows-style backslashes with
forward slashes. This normalization ensures that the path can be used consistently across different operating
systems. The function returns a Path object representing the normalized path.

Arguments:
- path_text: A string representing the file path to normalize.

Returns:
- A Path object representing the normalized path.
'''
def normalize_path(path_text: str) -> Path:
    # Metadata currently stores Windows separators; normalize for portability.
    return Path(path_text.replace("\\", "/"))

'''
# ----------------------------------------------------------------------------------------------------
# resolve_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function resolves a given path string to an absolute Path object. It first normalizes the input path
using the normalize_path function. If the normalized path is already absolute, it is returned as is. If the
normalized path is relative, the function first checks if it exists when resolved against the current working
directory. If it exists, that path is returned. Otherwise, the function resolves the path against a provided
root directory (such as the project root) and returns the resolved absolute path.

Arguments:
- path_text: A string representing the file path to resolve.
- root: A Path object representing the root directory to resolve against if the path is relative.

Returns:
- A Path object representing the resolved absolute path.
'''
def resolve_path(path_text: str, root: Path) -> Path:
    raw_path = normalize_path(path_text)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (root / raw_path).resolve()

'''
# ----------------------------------------------------------------------------------------------------
# stable_sample_seed()
# ----------------------------------------------------------------------------------------------------
Description:
This function generates a stable sample seed based on a global seed, source path, and augmentation index.
It creates a unique key by combining the global seed, source path, and augmentation index, and then computes
a SHA-256 hash of the key. The function returns an integer seed derived from the first 8 characters of the
hash digest, ensuring that the same input parameters will always produce the same sample seed. This allows for
deterministic augmentation results across different runs of the script, as long as the input parameters remain
the same.

Arguments:
- global_seed: An integer representing the global random seed for the augmentation process.
- source_path: A string representing the path of the source image being augmented.
- aug_index: An integer representing the index of the augmentation for the given source image.

Returns:
- An integer representing the stable sample seed derived from the input parameters.
'''
def stable_sample_seed(global_seed: int, source_path: str, aug_index: int) -> int:
    key = f"{global_seed}|{source_path}|{aug_index}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)

'''
# ----------------------------------------------------------------------------------------------------
# pairwise()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a list of values and returns a list of tuples, where each tuple contains a pair
of consecutive values from the input list. The function assumes that the input list has an even number
of elements, and it pairs the first element with the second, the third with the fourth, and so on. This
is particularly useful for processing flattened lists of parameters, such as translation offsets or noise
parameters, where each pair of values represents a related set of parameters (e.g., x and y offsets or
mean and stddev for noise). The function returns a list of tuples that can be easily used in the augmentation
functions.

Arguments:
- values: A list of values to be paired. The list should have an even number of elements.

Returns:
- A list of tuples, where each tuple contains a pair of consecutive values from the input list.
'''
def pairwise(values: List[float]) -> List[Tuple[float, float]]:
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

'''
# ----------------------------------------------------------------------------------------------------
# load_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function loads and filters rows from the input metadata CSV file. It reads the CSV file using a 
DictReader and iterates through each row, applying filters to keep only those rows that are marked as
"train" in the "split" column, have a status of "OK" in the "status" column, and contain a non-empty
"path" column. The function collects the valid rows in a list and returns it. If no valid training rows
are found, the function raises a ValueError indicating that no valid training rows were found in the 
metadata.

Arguments:
- metadata_csv: A Path object representing the path to the input metadata CSV file.

Returns:
- A list of dictionaries, where each dictionary represents a valid row from the metadata CSV file that
  meets the specified filtering criteria.

Raises:
- ValueError: If no valid training rows are found in the metadata CSV file.
'''
def load_rows(metadata_csv: Path) -> List[Dict[str, str]]:
    rows = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Keep train-only for v1 to avoid leakage.
            if row.get("split", "").strip().lower() != "train":
                continue
            if row.get("status", "").strip().upper() != "OK":
                continue
            if not row.get("path"):
                continue
            rows.append(row)
    if not rows:
        raise ValueError("No valid training rows found in metadata.")
    return rows

'''
# ----------------------------------------------------------------------------------------------------
# resolve_source_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function resolves the source image path from the metadata. It first normalizes the input path text
using the normalize_path function. If the normalized path is already absolute, it is returned as is. If
the normalized path is relative, the function first checks if it exists when resolved against the current
working directory. If it exists, that path is returned. Otherwise, the function resolves the path against
the project root directory and returns the resolved absolute path. This function ensures that the source
image can be located correctly regardless of whether the path in the metadata is absolute or relative, and
it provides flexibility in how the paths are specified in the metadata CSV file.

Arguments:
- path_text: A string representing the file path of the source image as specified in the metadata.

Returns:
- A Path object representing the resolved absolute path to the source image.

Raises:
- FileNotFoundError: If the source image cannot be found at the resolved path.
'''
def resolve_source_path(path_text: str) -> Path:
    raw_path = normalize_path(path_text)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PROJECT_ROOT / raw_path).resolve()

'''
# ----------------------------------------------------------------------------------------------------
# rotation_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function rotates the input image by a specified angle. It uses the PIL library's rotate function
to perform the rotation, applying bicubic resampling for better quality and expanding the output image
to fit the entire rotated image. The function takes an Image object and a float representing the angle
of rotation in degrees, and it returns a new Image object that is the result of the rotation transformation.

Arguments:
- image: A PIL Image object representing the input image to be rotated.
- angle: A float representing the angle of rotation in degrees. Positive values rotate the image counter
         clockwise, while negative values rotate it clockwise.

Returns:
- A new PIL Image object that is the result of rotating the input image by the specified angle
'''
def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

'''
# ----------------------------------------------------------------------------------------------------
# flip_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function flips the input image either horizontally or vertically based on the specified mode. It 
uses the PIL library's ImageOps module to perform the flipping transformation. If the mode is "horizontal",
the function applies a horizontal flip (mirror) to the image. If the mode is "vertical", the function applies
a vertical flip (top-to-bottom) to the image. If an invalid mode is provided, the function raises a ValueError
indicating that the flip mode is invalid. The function takes an Image object and a string representing the flip
mode, and it returns a new Image object that is the result of the flipping transformation.

Arguments:
- image: A PIL Image object representing the input image to be flipped.
- mode: A string representing the flip mode. Valid values are "horizontal" for a horizontal flip and "vertical"
        for a vertical flip.

Returns:
- A new PIL Image object that is the result of flipping the input image according to the specified mode.

Raises:
- ValueError: If the provided flip mode is not "horizontal" or "vertical".
'''
def flip_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "horizontal":
        return ImageOps.mirror(image)
    elif mode == "vertical":
        return ImageOps.flip(image)
    else:
        raise ValueError(f"Invalid flip mode: {mode}")

'''
# ----------------------------------------------------------------------------------------------------
# scale_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function scales the input image by a specified factor. It calculates the new width and height of 
the image by multiplying the original dimensions by the scaling factor. The function then uses the PIL
library's resize function to perform the scaling transformation, applying bicubic resampling for better
quality. The function takes an Image object and a float representing the scaling factor, and it returns
a new Image object that is the result of scaling the input image by the specified factor.

Arguments:
- image: A PIL Image object representing the input image to be scaled.
- factor: A float representing the scaling factor. Values greater than 1 will enlarge the image, while
          values between 0 and 1 will reduce the size of the image.

Returns:
- A new PIL Image object that is the result of scaling the input image by the specified factor.
'''
def scale_image(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    return image.resize((new_width, new_height), resample=Image.BICUBIC)

'''
# ----------------------------------------------------------------------------------------------------
# translate_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function translate the input image by specified x and y offsets. It uses the PIL library's ImageChops
module to perform the translation transformation. The function takes an Image object and two integers
representing the x and y offsets for the translation. The x_offset specifies how many pixels to shift the 
image horizontally (positive values shift right, negative values shift left), while the y_offset specifies
how many pixels to shift the image vertically (positive values shift down, negative values shift up). The
function returns a new Image object that is the result of translating the input image by the specified offsets.

Arguments:
- image: A PIL Image object representing the input image to be translated.
- x_offset: An integer representing the number of pixels to shift the image horizontally. Positive values
            shift the image to the right, while negative values shift it to the left.
- y_offset: An integer representing the number of pixels to shift the image vertically. Positive values
            shift the image down, while negative values shift it up.

Returns:
- A new PIL Image object that is the result of translating the input image by the specified x and y offsets.
'''
def translate_image(image: Image.Image, x_offset: int, y_offset: int) -> Image.Image:
    return ImageChops.offset(image, x_offset, y_offset)

'''
# ----------------------------------------------------------------------------------------------------
# adjust_brightness()
# ----------------------------------------------------------------------------------------------------
Description:
This function adjusts the brightness of the input image by a specified factor. It uses the PIL library's
ImageEnhance module to perform the brightness adjustment transformation. The function takes an Image 
object and a float representing the brightness factor. A factor of 1.0 means no change, values less than
1.0 will darken the image, and values greater than 1.0 will brighten the image. The function returns a 
new Image object that is the result of adjusting the brightness of the input image by the specified factor.

Arguments:
- image: A PIL Image object representing the input image whose brightness is to be adjusted.
- factor: A float representing the brightness adjustment factor. A value of 1.0 means no change, values
          less than 1.0 will darken the image, and values greater than 1.0 will brighten the image.

Returns:
- A new PIL Image object that is the result of adjusting the brightness of the input image by the specified factor.
'''
def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

'''
# ----------------------------------------------------------------------------------------------------
# adjust_contrast()
# ----------------------------------------------------------------------------------------------------
Description:
This function adjusts the contrast of the input image by a specified factor. It uses the PIL library's
ImageEnhance module to perform the contrast adjustment transformation. The functiont takes an Image 
object and a float representing the contrast factor. A factor of 1.0 means no change, values less than
1.0 will reduce the contrast of the image, and values greater than 1.0 will increase the contrast of the
image. The function returns a new Image object that is the result of adjusting the contrast of the input
image by the specified factor.

Arguments:
- image: A PIL Image object representing the input image whose contrast is to be adjusted.
- factor: A float representing the contrast adjustment factor. A value of 1.0 means no change, values
          less than 1.0 will reduce the contrast of the image, and values greater than 1.0 will increase
          the contrast of the image.

Returns:
- A new PIL Image object that is the result of adjusting the contrast of the input image by the specified factor.
'''
def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

'''
# ----------------------------------------------------------------------------------------------------
# add_noise()
# ----------------------------------------------------------------------------------------------------
Description:
This function adds random Gaussian noise to the input image based on a specified mean and standard
deviation. It first checks if the provided standard deviation is non-negative, as negative values are
not valid for noise generation. The function then determines the mode for the noisy image (grayscale "L"
or color "RGB") based on the mode of the input image. It creates a copy of the input image in the appropriate
mode and converts it to a byte array for pixel manipulation. The function iterates through each pixel value
in the byte array, adding random noise generated from a Gaussian distribution with the specified mean and
standard deviation. The resulting noisy pixel value is clamped to the valid range of [0, 255] to ensure it
remains a valid pixel intensity. Finally, the function creates and returns a new Image object from the 
modified byte array, representing the input image with added Gaussian noise. The function takes an Image
object, a float representing the mean of the noise, a float representing the standard deviation of the noise,
and a random.Random object for generating the random noise values.

Arguments:
- image: A PIL Image object representing the input image to which noise will be added.
- mean: A float representing the mean of the Gaussian noise to be added to the image.
- stddev: A float representing the standard deviation of the Gaussian noise to be added to the image. 
          Must be non-negative.
- rng: A random.Random object used to generate random noise values from a Gaussian distribution.

Returns:
- A new PIL Image object that is the result of adding Gaussian noise to the input image based on the 
  specified mean and standard deviation.

Raises:
- ValueError: If the provided standard deviation for the noise is negative.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# random_crop()
# ----------------------------------------------------------------------------------------------------
Description:
This function performs a random crop augmentation on the input image based on a specified crop size. 
It first checks if the provided crop size is smaller than the original image size, as cropping to a 
larger size is not valid. The function then calculates the maximum possible x and y offsets for the
crop based on the original image dimensions and the specified crop size. It generates random x and y
offsets within the valid range using the provided random.Random object. Finally, the function uses
the PIL library's crop function to extract and return the cropped region of the image based on the 
calculated offsets and the specified crop size. The function takes an Image object, a tuple
representing the crop size (width, height), and a random.Random object for generating the random crop
offsets.

Arguments:
- image: A PIL Image object representing the input image to be cropped.
- crop_size: A tuple (crop_width, crop_height) representing the width and height of the crop. Both values
             must be smaller than the corresponding dimensions of the input image.
- rng: A random.Random object used to generate random x and y offsets for the crop within the valid range.

Returns:
- A new PIL Image object that is the result of cropping the input image to the specified crop size at a
  random location determined by the generated offsets.

Raises:
- ValueError: If the provided crop size is larger than the original image size.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# random_erasing()
# ----------------------------------------------------------------------------------------------------
Description:
This function performs a random erasing augmentation on the input image by filling a randomly selected
rectangular region with a specified pixel value. It first checks if the provided erase size is smaller 
than the original image size, as erasing a larger area is not valid. The function then calculates the
maximum possible x and y offsets for the erase region based on the original image dimensions and the
specified erase size. It generates random x and y offsets within the valid range using the provided
random.Random object. The function creates a copy of the input image and uses the PIL library's ImageDraw
module to draw a filled rectangle on the copied image at the calculated offsets with the specified erase
size and pixel value. Finally, the function returns the modified image with the random erasing applied.
The function takes an Image object, a tuple representing the erase size (width, height), an integer
representing the pixel value to fill the erased region, and a random.Random object for generating the
random erase offsets.

Arguments:
- image: A PIL Image object representing the input image to which random erasing will be applied.
- erase_size: A tuple (erase_width, erase_height) representing the width and height of the rectangular region
              to be erased. Both values must be smaller than the corresponding dimensions of the input image.
- erase_value: An integer representing the pixel value to fill the erased region. Must be between 0 and 255.
- rng: A random.Random object used to generate random x and y offsets for the erase region within the valid range.

Returns:
- A new PIL Image object that is the result of applying random erasing to the input image by filling a randomly
  selected rectangular region with the specified pixel value.

Raises:
- ValueError: If the provided erase size is larger than the original image size, or if the erase value is
              not between 0 and 255.
'''
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

# Callable type for augmenter function
AugmenterFn = Callable[[Image.Image, random.Random], Tuple[Image.Image, Dict[str, object]]]

'''
# ----------------------------------------------------------------------------------------------------
# build_augmenters()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a list of augmenter functions based on the command-line arguments provided to the
script. It first processes the translation offsets and noise parameters from the arguments to create pairs
of related parameters (e.g., x and y offsets for translation, mean and stddev for noise). The function then
checks which augmentation types are enabled based on the specific flags provided in the arguments, as well
as the use_all_augs flag. For each enabled augmentation type, the function defines a corresponding augmenter
function that takes an image and a random.Random object as input and returns the augmented image along with
a dictionary of the parameters used for that augmentation. The augmenter functions are defined as closures
that capture the relevant parameters from the arguments. The function collects the enabled augmenter functions
ina a list of tuples, where each tuple contains the name of the augmentation and the corresponding augmenter
function. Finally, the function returns the list of enabled augmenter functions that can be used for augmenting 
the images in the dataset.

Arguments:
- args: The parsed command-line arguments that specify which augmentations to enable and the parameters for 
        each augmentation.

Returns:
- A list of tuples, where each tuple contains the name of an enabled augmentation and a corresponding augmenter
  function that can be called to apply that augmentation to an image.

Raises:
- ValueError: If no augmentations are enabled based on the provided arguments, the function raises a ValueError
              indicating that at least one augmentation must be enabled.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# save_augmented_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function saves the augmented image to the specified output path. It first ensures that the parent
directories of the output path exist, creating them if necessary. The function then checks the mode of
the input image and converts it to "RGB" if it is not already in that mode, as JPEG format requires RGB
images. Finally, the function saves the image to the output path in JPEG format with a specified quality
level.

Arguments:
- image: A PIL Image object representing the augmented image to be saved.
- output_path: A Path object representing the file path where the augmented image should be saved.

Returns:
- None. The function saves the image to the specified output path and does not return any value.
'''
def save_augmented_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_to_save = image.convert("RGB") if image.mode != "RGB" else image
    image_to_save.save(output_path, format="JPEG", quality=95)

'''
# ----------------------------------------------------------------------------------------------------
# write_metadata_row()
# ----------------------------------------------------------------------------------------------------
Description:
This function writes a single row of metadata to the CSV file using the provided DictWriter. It takes a
dictionary representing the metadata for a single augmented image and writes it to the CSV file, ensuring
that the values are properly aligned with the defined OUTPUT_COLUMNS. The function uses a dictionary 
comprehension to create a new dictionary that contains only the keys specified in OUTPUT_COLUMNS, filling
in any missing keys with an empty string. This ensures that the metadata row is complete and consistent with
the expected format of the output metadata CSV file.

Arguments:
- writer: A csv.DictWriter object that is used to write the metadata row to the CSV file.
- payload: A dictionary containing the metadata information for a single augmented image. The keys in the
           dictionary should correspond to the columns defined in OUTPUT_COLUMNS, and the values should be the
           corresponding metadata values for that augmented image.

Returns:
- None. The function writes the metadata row to the CSV file using the provided DictWriter 
  and does not return any value.
'''
def write_metadata_row(writer: csv.DictWriter, payload: Dict[str, object]) -> None:
    writer.writerow({column: payload.get(column, "") for column in OUTPUT_COLUMNS})

'''
# ----------------------------------------------------------------------------------------------------
# augment_single_image()
# ----------------------------------------------------------------------------------------------------
Description:
This function performs the augmentation process for a single source image. It takes the source image
and iterates through the specified number of augmentations to be generated for that image. For each
augmentation, it generates a stable sample seed based on the global seed, source path, and augmentation
index, and uses that seed to create a random.Random object for deterministic augmentation. The function
then randomly selects an augmenter from the list of enabled augmenters and applies it to the source image,
saving the augmented image to the output directory with a filename that includes the source stem, augmentation
index, augmenter name, and sample seed. The function also writes a metadata row for each augmentation attempt,
indicating whether the augmentation was successful, failed or skipped. The function keeps track of the number
of successful, failed, and skipped augmentations for the given source image and returns these counts at the end 
of the process.

Arguments:
- source_image: A PIL Image object representing the source image to be augmented.
- source_path_text: A string representing the original path of the source image as specified in the 
                    metadata.
- label: A string representing the label associated with the source image, which will be included 
         in the metadata for the augmented images.
- split: A string representing the data split (e.g., "train", "val", "test") associated with the
         source image, which will be included in the metadata for the augmented images.
- augmenters: A list of tuples, where each tuple contains the name of an enabled augmentation
              and a corresponding augmenter function that can be called to apply that augmentation to the source image.
- num_augs_per_image: An integer representing the number of augmented images to generate for the given source image.
- output_image_dir: A Path object representing the directory where the augmented images should be saved.
- global_seed: An integer representing the global seed for deterministic augmentation, 
               which will be used to generate stable sample seeds for each augmentation.
- writer: A csv.DictWriter object that is used to write the metadata rows for each augmentation 
          attempt to the output metadata CSV file.
- overwrite: A boolean flag indicating whether to overwrite existing augmented images if they already exist at the
             output path. If False, the function will skip augmentation for images that already exist and write
             a metadata row indicating that the augmentation was skipped.

Returns:
- A tuple containing three integers: (success_count, fail_count, skip_count), where success_count is 
  the number of successful augmentations, fail_count is the number of failed augmentations, and skip_count is
  the number of skipped augmentations for the given source image.
'''
def augment_single_image(
    source_image: Image.Image,
    source_path_text: str,
    label: str,
    split: str,
    augmenters: List[Tuple[str, AugmenterFn]],
    num_augs_per_image: int,
    output_image_dir: Path,
    global_seed: int,
    writer: csv.DictWriter,
    overwrite: bool,
) -> Tuple[int, int, int]:
    success_count = 0
    fail_count = 0
    skip_count = 0

    source_stem = normalize_path(source_path_text).stem

    for aug_idx in range(num_augs_per_image):
        sample_seed = stable_sample_seed(global_seed, source_path_text, aug_idx)
        rng = random.Random(sample_seed)

        aug_name, aug_fn = rng.choice(augmenters)
        output_name = f"{source_stem}__aug{aug_idx:03d}__{aug_name}__s{sample_seed:08x}.jpg"
        output_path = output_image_dir / output_name

        if output_path.exists() and not overwrite:
            write_metadata_row(
                writer,
                {
                    "source_path": source_path_text,
                    "augmented_path": output_path.as_posix(),
                    "label": label,
                    "split": split,
                    "augmentation": aug_name,
                    "params_json": "{}",
                    "seed": sample_seed,
                    "status": "SKIPPED",
                    "error": "file exists and overwrite is disabled",
                },
            )
            skip_count += 1
            continue

        try:
            augmented_image, params = aug_fn(source_image.copy(), rng)
            save_augmented_image(augmented_image, output_path)

            write_metadata_row(
                writer,
                {
                    "source_path": source_path_text,
                    "augmented_path": output_path.as_posix(),
                    "label": label,
                    "split": split,
                    "augmentation": aug_name,
                    "params_json": json.dumps(params, ensure_ascii=True),
                    "seed": sample_seed,
                    "status": "OK",
                    "error": "",
                },
            )
            success_count += 1
        except Exception as exc:
            write_metadata_row(
                writer,
                {
                    "source_path": source_path_text,
                    "augmented_path": "",
                    "label": label,
                    "split": split,
                    "augmentation": aug_name,
                    "params_json": "{}",
                    "seed": sample_seed,
                    "status": "FAILED",
                    "error": str(exc),
                },
            )
            fail_count += 1

    return success_count, fail_count, skip_count

'''
# ----------------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------------
Description:
This is the main execution function for the data augmentation script. It sets up logging, parse command
line arguments, and resolves input and output paths. The function validates the provided arguments and 
builds the list of enabled augmenters based on the arguments. It then loads the rows from the input 
metadata CSV file and prepares the output directories. The function iterates through each row of the input
metadata, attempts to load the source image, and applies the specified augmentations while writing the
results to the output metadata CSV file. It keeps track of the number of successful, failed, and skipped
augmentations and logs the progress throughout the execution. Finally, it logs a summary of the augmentation
results, including the total number of successful, failed, and skipped augmentations.

Arguments:
- None. The function does not take any arguments and relies on command-line arguments for configuration.

Returns:
- None. The function performs the data augmentation process and writes the results to the specified output
  directories and files, but does not return any value.
'''
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

    rows = load_rows(input_metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    metadata_mode = "w" if args.overwrite or not output_metadata_path.exists() else "a"
    write_header = metadata_mode == "w"

    logging.info("Loaded %d training rows from %s", len(rows), input_metadata_path)
    logging.info("Enabled augmenters: %s", ", ".join(name for name, _ in augmenters))
    logging.info("Writing augmented images to %s", output_image_dir)
    logging.info("Writing metadata to %s", output_metadata_path)

    total_ok = 0
    total_failed = 0
    total_skipped = 0

    with open(output_metadata_path, mode=metadata_mode, newline="", encoding="utf-8") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            source_path_text = row.get("path", "")
            source_path = resolve_source_path(source_path_text)

            if not source_path.exists():
                logging.warning("Source image not found: %s", source_path)
                for aug_idx in range(args.num_augs_per_image):
                    sample_seed = stable_sample_seed(args.seed, source_path_text, aug_idx)
                    write_metadata_row(
                        writer,
                        {
                            "source_path": source_path_text,
                            "augmented_path": "",
                            "label": row.get("label", ""),
                            "split": row.get("split", ""),
                            "augmentation": "",
                            "params_json": "{}",
                            "seed": sample_seed,
                            "status": "FAILED",
                            "error": f"Source image not found: {source_path}",
                        },
                    )
                    total_failed += 1
                continue

            try:
                with Image.open(source_path) as source_image_handle:
                    source_image = ImageOps.exif_transpose(source_image_handle).convert("RGB")

                ok_count, fail_count, skip_count = augment_single_image(
                    source_image=source_image,
                    source_path_text=source_path_text,
                    label=row.get("label", ""),
                    split=row.get("split", ""),
                    augmenters=augmenters,
                    num_augs_per_image=args.num_augs_per_image,
                    output_image_dir=output_image_dir,
                    global_seed=args.seed,
                    writer=writer,
                    overwrite=args.overwrite,
                )

                total_ok += ok_count
                total_failed += fail_count
                total_skipped += skip_count
            except Exception as exc:
                logging.exception("Unexpected failure on source image %s", source_path)
                for aug_idx in range(args.num_augs_per_image):
                    sample_seed = stable_sample_seed(args.seed, source_path_text, aug_idx)
                    write_metadata_row(
                        writer,
                        {
                            "source_path": source_path_text,
                            "augmented_path": "",
                            "label": row.get("label", ""),
                            "split": row.get("split", ""),
                            "augmentation": "",
                            "params_json": "{}",
                            "seed": sample_seed,
                            "status": "FAILED",
                            "error": str(exc),
                        },
                    )
                    total_failed += 1

            if idx % 100 == 0:
                logging.info("Processed %d/%d source rows", idx, len(rows))

    logging.info(
        "Augmentation complete. Success=%d Failed=%d Skipped=%d",
        total_ok,
        total_failed,
        total_skipped,
    )

# Main entry point
if __name__ == "__main__":
    run()
