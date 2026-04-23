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

def string_value(value):
	if value is None:
		return ""
	return str(value).strip()

def normalize_split(value):
	text = string_value(value).lower()
	return text if text in SPLITS else ""

def normalize_int_text(value):
	text = string_value(value)
	if not text:
		return ""
	try:
		return str(int(float(text)))
	except ValueError:
		return text

def numeric_sort_key(value):
	text = string_value(value)
	if not text:
		return (2, 0, "")
	try:
		return (0, int(text), "")
	except ValueError:
		try:
			return (0, int(float(text)), "")
		except ValueError:
			return (1, 0, text)

def row_score(row, columns):
	return sum(1 for column in columns if string_value(row.get(column)))

def make_backup(csv_path):
	if not csv_path.exists():
		return None

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	backup_path = csv_path.parent / f"{csv_path.name}.bak_{timestamp}"
	shutil.copy2(csv_path, backup_path)
	return backup_path

def read_csv_rows(csv_path):
	if not csv_path.exists():
		logging.info("CSV not found, skipping read: %s", csv_path)
		return []

	with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		if not reader.fieldnames:
			logging.warning("CSV has no header, treating as empty: %s", csv_path)
			return []
		return list(reader)

def parse_street_filename(file_name):
	match = STREET_FILENAME_PATTERN.fullmatch(file_name)
	if not match:
		return None

	groups = match.groupdict()
	return {
		"id": string_value(groups["id"]),
		"split": normalize_split(groups["split"]),
		"heading": normalize_int_text(groups["heading"]),
		"pitch": normalize_int_text(groups["pitch"]),
		"fov": normalize_int_text(groups["fov"]),
		"ext": string_value(groups["ext"]).lower(),
	}

def parse_map_filename(file_name):
	match = MAP_FILENAME_PATTERN.fullmatch(file_name)
	if not match:
		return None

	groups = match.groupdict()
	return {
		"id": string_value(groups["id"]),
		"split": normalize_split(groups["split"]),
		"zoom": normalize_int_text(groups["zoom"]),
		"tile_x": normalize_int_text(groups["tile_x"]),
		"tile_y": normalize_int_text(groups["tile_y"]),
		"ext": string_value(groups["ext"]).lower(),
	}

def parse_filename(dataset_name, file_name):
	if dataset_name == "street":
		return parse_street_filename(file_name)
	if dataset_name == "map":
		return parse_map_filename(file_name)
	return None

def scan_split_files(config, base_dir):
	files_by_key = {}
	parse_errors = []
	duplicate_keys = []
	files_scanned = 0

	for split_name in SPLITS:
		split_dir = base_dir / Path(config.split_dirs[split_name])
		if not split_dir.exists():
			logging.warning("Split directory not found: %s", split_dir)
			continue

		if not split_dir.is_dir():
			logging.warning("Split path is not a directory: %s", split_dir)
			continue

		for file_path in sorted(split_dir.iterdir(), key=lambda path: path.name.lower()):
			if not file_path.is_file():
				continue

			if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
				continue

			files_scanned += 1
			parsed = parse_filename(config.name, file_path.name)
			if parsed is None:
				parse_errors.append(str(file_path))
				continue

			if parsed["split"] and parsed["split"] != split_name:
				logging.warning(
					"Filename split mismatch (%s != %s): %s",
					parsed["split"],
					split_name,
					file_path,
				)

			parsed["split"] = split_name
			key = build_key_from_parts(config.name, parsed)
			if key is None:
				parse_errors.append(str(file_path))
				continue

			if key in files_by_key:
				duplicate_keys.append((key, str(file_path), files_by_key[key]["abs_path"]))
				continue

			files_by_key[key] = {
				"parts": parsed,
				"file_name": file_path.name,
				"abs_path": str(file_path),
				"metadata_path": build_metadata_path(config, split_name, file_path.name),
			}

	return {
		"files_by_key": files_by_key,
		"files_scanned": files_scanned,
		"parse_errors": parse_errors,
		"duplicate_keys": duplicate_keys,
	}


def reconcile_dataset(config, base_dir, dry_run, keep_nonfile_rows, backup_enabled):
	combined_path = base_dir / Path(config.combined_metadata)
	split_paths = {
		split_name: base_dir / Path(config.split_metadata[split_name]) for split_name in SPLITS
	}

	combined_rows_existing = read_csv_rows(combined_path)
	split_rows_existing = []
	for split_name in SPLITS:
		split_rows_existing.extend(read_csv_rows(split_paths[split_name]))

	all_existing_rows = combined_rows_existing + split_rows_existing

	scan_result = scan_split_files(config, base_dir)
	files_by_key = scan_result["files_by_key"]

	context_by_key, invalid_file_rows = build_context_by_key(config, all_existing_rows)
	id_context = build_id_context(config, all_existing_rows)

	actual_keys = set(files_by_key.keys())
	existing_keys = set(context_by_key.keys())

	file_rows = []
	for key, file_info in files_by_key.items():
		key_row = context_by_key.get(key)
		id_row = id_context.get(file_info["parts"]["id"])
		file_rows.append(build_row_for_file(config, file_info, key_row, id_row))

	nonfile_rows = []
	if keep_nonfile_rows:
		for row in combined_rows_existing:
			if string_value(row.get("path", "")):
				continue

			normalized = normalize_row(row, config.columns)
			if normalize_split(normalized.get("split", "")) not in SPLITS:
				continue

			nonfile_rows.append(normalized)

		nonfile_rows = dedupe_rows(nonfile_rows, config.columns)

	combined_output_rows = sort_rows(config, file_rows + nonfile_rows)

	split_output_rows = {}
	for split_name in SPLITS:
		split_rows = [
			row
			for row in combined_output_rows
			if normalize_split(row.get("split", "")) == split_name
		]
		split_output_rows[split_name] = sort_rows(config, split_rows)

	backups = []
	if not dry_run:
		write_csv_rows(combined_path, config.columns, combined_output_rows, backup_enabled, backups)
		for split_name in SPLITS:
			write_csv_rows(
				split_paths[split_name],
				config.columns,
				split_output_rows[split_name],
				backup_enabled,
				backups,
			)

	if scan_result["parse_errors"]:
		logging.warning(
			"%s | unparseable filenames: %s",
			config.name,
			len(scan_result["parse_errors"]),
		)
		for sample in scan_result["parse_errors"][:10]:
			logging.warning("%s | parse error sample: %s", config.name, sample)

	if scan_result["duplicate_keys"]:
		logging.warning(
			"%s | duplicate file keys found: %s",
			config.name,
			len(scan_result["duplicate_keys"]),
		)

	return {
		"dataset": config.name,
		"dry_run": dry_run,
		"files_scanned": scan_result["files_scanned"],
		"files_indexed": len(files_by_key),
		"parse_errors": len(scan_result["parse_errors"]),
		"duplicate_file_keys": len(scan_result["duplicate_keys"]),
		"existing_file_keys": len(existing_keys),
		"actual_file_keys": len(actual_keys),
		"kept_file_keys": len(existing_keys & actual_keys),
		"added_file_keys": len(actual_keys - existing_keys),
		"removed_file_keys": len(existing_keys - actual_keys),
		"invalid_existing_file_rows": invalid_file_rows,
		"preserved_nonfile_rows": len(nonfile_rows),
		"combined_output_rows": len(combined_output_rows),
		"split_output_rows": {
			split_name: len(split_output_rows[split_name]) for split_name in SPLITS
		},
		"combined_metadata_path": str(combined_path),
		"split_metadata_paths": {
			split_name: str(split_paths[split_name]) for split_name in SPLITS
		},
		"backups": backups,
	}

def log_dataset_summary(summary):
	logging.info("Dataset: %s", summary["dataset"])
	logging.info("Dry run: %s", summary["dry_run"])
	logging.info(
		"Files scanned=%s indexed=%s parse_errors=%s duplicate_keys=%s",
		summary["files_scanned"],
		summary["files_indexed"],
		summary["parse_errors"],
		summary["duplicate_file_keys"],
	)
	logging.info(
		"Existing keys=%s actual keys=%s kept=%s added=%s removed=%s invalid_existing_rows=%s",
		summary["existing_file_keys"],
		summary["actual_file_keys"],
		summary["kept_file_keys"],
		summary["added_file_keys"],
		summary["removed_file_keys"],
		summary["invalid_existing_file_rows"],
	)
	logging.info(
		"Preserved non-file rows=%s combined output rows=%s",
		summary["preserved_nonfile_rows"],
		summary["combined_output_rows"],
	)
	logging.info("Split output rows: %s", summary["split_output_rows"])
	logging.info("Combined metadata: %s", summary["combined_metadata_path"])
	logging.info("Split metadata: %s", summary["split_metadata_paths"])
	if summary["backups"]:
		logging.info("Backup files created: %s", len(summary["backups"]))

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

	for dataset_name in selected:
		config = DATASET_CONFIGS[dataset_name]
		summary = reconcile_dataset(
			config=config,
			base_dir=base_dir,
			dry_run=args.dry_run,
			keep_nonfile_rows=keep_nonfile_rows,
			backup_enabled=backup_enabled,
		)
		log_dataset_summary(summary)

	logging.info("Metadata reconciliation complete.")

if __name__ == "__main__":
	run()
	