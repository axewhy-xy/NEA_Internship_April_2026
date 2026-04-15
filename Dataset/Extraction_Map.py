"""
Dataset extraction script for Google Maps Tile images.

Author: AxeWhy
Date: 2026-04-15
Last updated: 2026-04-15

Description:
Builds a Google Maps Tile image dataset from a CSV of locations and labels.
Supports both coordinate and postal-code input formats, performs
train/validation/test splitting, downloads images using the Satellite Tile API 
at multiple zoom levels, and writes detailed metadata logs.

Input CSV formats supported:
1) id,lat,lon,label
2) id,postal_code,label

Usage example:
python Dataset/Extraction_Map.py --input-csv Dataset/hdb_points_postal.csv

API key:
Set GOOGLE_MAPS_API_KEY in your environment, or pass --api-key.
"""

import argparse
import csv
import datetime
import logging
import math
import os
import random
import re
import time
from pathlib import Path

import requests

# ----------------------------------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------------------------------
DEFAULT_INPUT_CSV                                       = "Dataset/Input/hdb_points_postal.csv"
DEFAULT_OUTPUT_ROOT                                     = Path("Dataset/Map")
DEFAULT_COMBINED_METADATA                               = "map_metadata.csv"

DEFAULT_RANDOM_SEED                                     = 42
DEFAULT_TRAIN_RATIO                                     = 0.7
DEFAULT_VALID_RATIO                                     = 0.15

DEFAULT_MAP_TYPE                                        = "satellite"
DEFAULT_ZOOM_LEVELS                                     = "20"
DEFAULT_LANGUAGE                                        = "en-US"
DEFAULT_REGION                                          = "SG"
DEFAULT_IMAGE_FORMAT                                    = "jpeg"

DEFAULT_REQUEST_TIMEOUT_SECONDS                         = 30
DEFAULT_MAX_RETRIES                                     = 4
DEFAULT_RETRY_BACKOFF_SECONDS                           = 1.2
DEFAULT_MIN_DELAY_SECONDS                               = 0.2
DEFAULT_GEOCODE_DELAY_SECONDS                           = 0.2
DEFAULT_SESSION_REFRESH_SECONDS                         = 60

MAX_TILE_ZOOM_LEVEL                                     = 22
RETRY_HTTP_STATUSES                                     = {429, 500, 502, 503, 504}
SESSION_REFRESH_HTTP_STATUSES                           = {400, 401, 403}

CREATE_SESSION_URL                                      = "https://tile.googleapis.com/v1/createSession"
TILE_URL_TEMPLATE                                       = "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"
ONEMAP_GEOCODE_URL                                      = "https://www.onemap.gov.sg/api/common/elastic/search"

SUPPORTED_MAP_TYPES                                     = {"satellite", "roadmap", "terrain"}
SUPPORTED_IMAGE_FORMATS                                 = {"jpeg", "png", "webp"}

METADATA_COLUMNS                                        = [
                                                            "id",
                                                            "lat",
                                                            "lon",
                                                            "postal_code",
                                                            "label",
                                                            "split",
                                                            "map_type",
                                                            "zoom",
                                                            "tile_x",
                                                            "tile_y",
                                                            "tile_width",
                                                            "tile_height",
                                                            "image_format",
                                                            "session_id",
                                                            "status",
                                                            "path",
                                                            "error",
                                                        ]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Extract Google Map Tiles dataset with split folders and metadata."
	)
	parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
	parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
	parser.add_argument("--metadata-csv", default=DEFAULT_COMBINED_METADATA)
	parser.add_argument("--api-key", default=os.getenv("GOOGLE_MAPS_API_KEY"))

	parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
	parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
	parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)

	parser.add_argument(
		"--map-type",
		default=DEFAULT_MAP_TYPE,
		help="Google Map Tiles map type: satellite, roadmap, or terrain.",
	)
	parser.add_argument(
		"--zoom-levels",
		default=DEFAULT_ZOOM_LEVELS,
		help="Comma-separated zoom levels, e.g. 19,20,21",
	)
	parser.add_argument("--language", default=DEFAULT_LANGUAGE)
	parser.add_argument("--region", default=DEFAULT_REGION)
	parser.add_argument(
		"--image-format",
		default=DEFAULT_IMAGE_FORMAT,
		help="Preferred tile format: jpeg, png, or webp.",
	)

	parser.add_argument(
		"--timeout-seconds", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
	)
	parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
	parser.add_argument(
		"--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS
	)
	parser.add_argument("--min-delay-seconds", type=float, default=DEFAULT_MIN_DELAY_SECONDS)
	parser.add_argument(
		"--geocode-delay-seconds", type=float, default=DEFAULT_GEOCODE_DELAY_SECONDS
	)
	parser.add_argument(
		"--session-refresh-seconds",
		type=int,
		default=DEFAULT_SESSION_REFRESH_SECONDS,
		help="Refresh tile session this many seconds before expiry.",
	)
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Skip download when target tile file already exists.",
	)
	parser.add_argument(
		"--metadata-only",
		action="store_true",
		help="Only compute tile coordinates and write metadata; skip downloads.",
	)
	return parser.parse_args()

def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

def validate_args(args):
	if not args.api_key:
		raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key.")
	if args.train_ratio <= 0 or args.valid_ratio < 0:
		raise ValueError("Split ratios must be non-negative and train > 0.")
	if args.train_ratio + args.valid_ratio >= 1.0:
		raise ValueError("train_ratio + valid_ratio must be < 1.0.")
	if args.timeout_seconds <= 0:
		raise ValueError("--timeout-seconds must be > 0.")
	if args.max_retries < 0:
		raise ValueError("--max-retries must be >= 0.")
	if args.retry_backoff_seconds <= 0:
		raise ValueError("--retry-backoff-seconds must be > 0.")
	if args.min_delay_seconds < 0:
		raise ValueError("--min-delay-seconds must be >= 0.")
	if args.geocode_delay_seconds < 0:
		raise ValueError("--geocode-delay-seconds must be >= 0.")
	if args.session_refresh_seconds < 0:
		raise ValueError("--session-refresh-seconds must be >= 0.")

	map_type = str(args.map_type).strip().lower()
	if map_type not in SUPPORTED_MAP_TYPES:
		raise ValueError(
			f"--map-type must be one of: {', '.join(sorted(SUPPORTED_MAP_TYPES))}."
		)

	image_format = normalize_image_format(args.image_format)
	if image_format not in SUPPORTED_IMAGE_FORMATS:
		raise ValueError(
			f"--image-format must be one of: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}."
		)

	# Validate zoom levels early for better UX.
	parse_zoom_levels(args.zoom_levels)

def parse_zoom_levels(zoom_levels_text):
	levels = []
	for token in zoom_levels_text.split(","):
		token = token.strip()
		if not token:
			continue

		level = int(token)
		if level < 0 or level > MAX_TILE_ZOOM_LEVEL:
			raise ValueError(f"Zoom level {level} out of range [0, {MAX_TILE_ZOOM_LEVEL}].")
		levels.append(level)

	if not levels:
		raise ValueError("At least one zoom level is required.")

	# Preserve insertion order while deduplicating.
	return list(dict.fromkeys(levels))	

def normalize_image_format(image_format):
	text = str(image_format or "").strip().lower()
	if text.startswith("image/"):
		text = text.split("/", 1)[1]
	if text == "jpg":
		text = "jpeg"
	return text

def ensure_output_dirs(output_root):
	train_dir = output_root / "Map_Training_Set"
	valid_dir = output_root / "Map_Validation_Set"
	test_dir = output_root / "Map_Test_Set"

	for directory in (train_dir, valid_dir, test_dir):
		directory.mkdir(parents=True, exist_ok=True)

	return train_dir, valid_dir, test_dir

def run():
	setup_logging()
	args = parse_args()
	validate_args(args)
	
	zoom_levels = parse_zoom_levels(args.zoom_levels)
	map_type = str(args.map_type).strip().lower()

	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	train_dir, valid_dir, test_dir = ensure_output_dirs(output_root)



if __name__ == "__main__":
	run()