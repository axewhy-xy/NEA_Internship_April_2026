"""
Update the metadata of the dataset.

Author: Axewhy
Date: 2026-04-23
Last Updated: 2026-04-23

Description:
This script reconciles metadata CSV files against the image files currently present in
the dataset folders. It supports both Street and Map datasets and updates:
1) Combined metadata CSV
2) Split metadata CSVs (train/valid/test)

For each dataset type, the script:
- Removes stale file rows (metadata rows whose image no longer exists)
- Adds missing file rows (image exists but metadata row is missing)
- Preserves existing row details whenever available (lat/lon/label/etc.)
- Optionally keeps non-file rows (for example, Street ZERO_RESULTS rows)

Usage:
python Dataset/Update_Metadata.py --dataset both
"""
import argparse
import csv
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------------------------------
SPLITS                                      = ("train", "valid", "test")
SPLIT_ORDER                                 = {"train": 0, "valid": 1, "test": 2}
SUPPORTED_IMAGE_EXTENSIONS                  = {".jpg", ".jpeg", ".png", ".webp"}

STREET_COLUMNS                              = (
                                            "id",
                                            "lat",
                                            "lon",
                                            "postal_code",
                                            "label",
                                            "split",
                                            "heading",
                                            "pitch",
                                            "fov",
                                            "size",
                                            "pano_id",
                                            "capture_date",
                                            "status",
                                            "path",
                                            "error",
                                        )

MAP_COLUMNS                                 = (
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
                                        )

STREET_FILENAME_PATTERN = re.compile(
	r"^(?P<id>.+)_(?P<split>train|valid|test)_h(?P<heading>-?\d+)_p(?P<pitch>-?\d+)_f(?P<fov>-?\d+)\.(?P<ext>[A-Za-z0-9]+)$",
	re.IGNORECASE,
)

MAP_FILENAME_PATTERN = re.compile(
	r"^(?P<id>.+)_(?P<split>train|valid|test)_z(?P<zoom>-?\d+)_x(?P<tile_x>-?\d+)_y(?P<tile_y>-?\d+)\.(?P<ext>[A-Za-z0-9]+)$",
	re.IGNORECASE,
)

@dataclass(frozen=True)
class DatasetConfig:
	name: str
	combined_metadata: str
	split_metadata: Dict[str, str]
	split_dirs: Dict[str, str]
	path_prefix: str
	columns: Tuple[str, ...]


DATASET_CONFIGS = {
	"street": DatasetConfig(
		name="street",
		combined_metadata="Dataset/Street/street_metadata.csv",
		split_metadata={
			"train": "Dataset/Street/street_metadata_train.csv",
			"valid": "Dataset/Street/street_metadata_valid.csv",
			"test": "Dataset/Street/street_metadata_test.csv",
		},
		split_dirs={
			"train": "Dataset/Street/Street_Training_Set",
			"valid": "Dataset/Street/Street_Validation_Set",
			"test": "Dataset/Street/Street_Test_Set",
		},
		path_prefix="Dataset/Street",
		columns=STREET_COLUMNS,
	),
	"map": DatasetConfig(
		name="map",
		combined_metadata="Dataset/Map/map_metadata.csv",
		split_metadata={
			"train": "Dataset/Map/map_metadata_train.csv",
			"valid": "Dataset/Map/map_metadata_valid.csv",
			"test": "Dataset/Map/map_metadata_test.csv",
		},
		split_dirs={
			"train": "Dataset/Map/Map_Training_Set",
			"valid": "Dataset/Map/Map_Validation_Set",
			"test": "Dataset/Map/Map_Test_Set",
		},
		path_prefix="Dataset/Map",
		columns=MAP_COLUMNS,
	),
}

def parse_args():
	parser = argparse.ArgumentParser(
		description="Reconcile Street/Map metadata CSVs with files on disk."
	)
	parser.add_argument(
		"--dataset",
		choices=["street", "map", "both"],
		default="both",
		help="Which dataset metadata to update.",
	)
	parser.add_argument(
		"--base-dir",
		default=".",
		help="Repository root directory containing Dataset/.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Run checks and summary only, do not write CSV files.",
	)
	parser.add_argument(
		"--drop-nonfile-rows",
		action="store_true",
		help="Drop rows with empty path (for example Street ZERO_RESULTS rows).",
	)
	parser.add_argument(
		"--no-backup",
		action="store_true",
		help="Do not create backup CSVs before writing updates.",
	)
	parser.add_argument(
		"--log-level",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		default="INFO",
	)
	return parser.parse_args()


def setup_logging(log_level):
	logging.basicConfig(
		level=getattr(logging, log_level),
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	
def run():
	args = parse_args()
	setup_logging(args.log_level)

	base_dir = Path(args.base_dir).resolve()
	keep_nonfile_rows = not args.drop_nonfile_rows
	backup_enabled = not args.no_backup

	if args.dataset == "both":
		selected = ["street", "map"]
	else:
		selected = [args.dataset]

	logging.info("Base directory: %s", base_dir)
	logging.info("Datasets selected: %s", selected)


if __name__ == "__main__":
	run()
	