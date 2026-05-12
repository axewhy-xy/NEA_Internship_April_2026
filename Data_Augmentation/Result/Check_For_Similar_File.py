"""
Check for similar files in the augmented dataset, sort duplicates, and update metadata.

Author: AxeWhy
Date: 2026-05-12
Last Updated: 2026-05-12

Description:
This script scans the augmented dataset image folders (train/valid/test/unknown), detects similar files
based on file-name heuristics, moves duplicates to the unknown folder, and reconciles the metadata CSV
so it matches the files on disk. It can optionally add missing metadata rows and sort the output.

Usage:
python Data_Augmentation/Result/Check_For_Similar_File.py --help
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ----------------------------------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------------------------------
PROJECT_ROOT                                        = Path(__file__).resolve().parents[2]

SUPPORTED_IMAGE_EXTENSIONS                          = {".jpg", ".jpeg", ".png", ".webp"}

SPLITS                                              = ("train", "valid", "test")
KNOWN_SPLITS                                        = ("train", "valid", "test", "unknown")
SPLIT_ORDER                                         = {"train": 0, "valid": 1, "test": 2, "unknown": 3}

OUTPUT_COLUMNS                                      = [
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

AUGMENTED_PATTERN = re.compile(
	r"^(?P<source>.+?)__aug(?P<index>\d{3})__(?P<aug>.+?)__s(?P<seed>[0-9a-fA-F]{8})\.(?P<ext>[A-Za-z0-9]+)$"
)
SPLIT_TOKEN_PATTERN = re.compile(r"_(train|valid|val|test)_", re.IGNORECASE)


@dataclass(frozen=True)
class FileRecord:
	path: Path
	split: str
	name: str
	similarity_key: str
	source_stem: str

'''
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
Description:
This function parses command-line arguments for the script. It defines various options that control 
the behavior of the script, such as the output directory, metadata file, image directory, similarity
mode, split preference, and logging level.

Arguments:
--output-dir: Base output directory containing images/ and augmented_metadata.csv (default: Data_Augmentation/Result).
--metadata: Metadata CSV filename or path (default: augmented_metadata.csv).
--images-dir: Images directory name or path (default: images).
--similarity-mode: How to detect similar files: filename (exact), source-stem (before __aug), 
                   source-id (strip split token from source-stem) (default: source-id).
--split-preference: Comma-separated preference order when resolving duplicates (default: train,valid,test).
--prefer-metadata-split: Prefer split from metadata when resolving duplicates (default: True).
--no-prefer-metadata-split: Ignore metadata split when resolving duplicates.
--no-add-orphans: Do not add metadata rows for files missing from the CSV (default: True).
--no-sort-metadata: Do not sort the output metadata rows (default: True).
--dry-run: Preview changes without moving files or writing metadata.
--no-backup: Do not create a backup of the metadata CSV before writing.
--log-level: Logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO).

Returns:
Namespace: Parsed command-line arguments.
'''
def parse_args():
	parser = argparse.ArgumentParser(
		description="Find similar augmented files, move duplicates to unknown, and update metadata."
	)
	parser.add_argument(
		"--output-dir",
		default="Data_Augmentation/Result",
		help="Base output directory containing images/ and augmented_metadata.csv.",
	)
	parser.add_argument(
		"--metadata",
		default="augmented_metadata.csv",
		help="Metadata CSV filename or path (default: augmented_metadata.csv).",
	)
	parser.add_argument(
		"--images-dir",
		default="images",
		help="Images directory name or path (default: images).",
	)
	parser.add_argument(
		"--similarity-mode",
		choices=["filename", "source-stem", "source-id"],
		default="source-id",
		help=(
			"How to detect similar files: filename (exact), source-stem (before __aug), "
			"source-id (strip split token from source-stem)."
		),
	)
	parser.add_argument(
		"--split-preference",
		default="train,valid,test",
		help="Comma-separated preference order when resolving duplicates.",
	)
	parser.add_argument(
		"--prefer-metadata-split",
		action="store_true",
		default=True,
		help="Prefer split from metadata when resolving duplicates (default).",
	)
	parser.add_argument(
		"--no-prefer-metadata-split",
		dest="prefer_metadata_split",
		action="store_false",
		help="Ignore metadata split when resolving duplicates.",
	)
	parser.add_argument(
		"--no-add-orphans",
		dest="add_orphans",
		action="store_false",
		default=True,
		help="Do not add metadata rows for files missing from the CSV.",
	)
	parser.add_argument(
		"--no-sort-metadata",
		dest="sort_metadata",
		action="store_false",
		default=True,
		help="Do not sort the output metadata rows.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview changes without moving files or writing metadata.",
	)
	parser.add_argument(
		"--no-backup",
		action="store_true",
		help="Do not create a backup of the metadata CSV before writing.",
	)
	parser.add_argument(
		"--log-level",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		default="INFO",
	)
	return parser.parse_args()

'''
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
Description:
This function sets up the logging configuration for the script. It configures the logging level,
format, and date format for log messages.

Arguments:
level (str): The logging level to use (e.g., "DEBUG", "INFO", "WARNING", "ERROR").

Returns:
None
'''
def setup_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level),
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

'''
# ----------------------------------------------------------------------------------------------------
# normalize_path_text()
# ----------------------------------------------------------------------------------------------------
Description:
This function normalizes a path string by replacing backslashes with forward slashes and converting it
to a Path object.

Arguments:
path_text (str): The input path string to normalize.

Returns:
Path: A Path object representing the normalized path.
'''
def normalize_path_text(path_text: str) -> Path:
	return Path(path_text.replace("\\", "/"))

'''
# ----------------------------------------------------------------------------------------------------
# resolve_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function resolves a given path text to an absolute Path object. If the input path is relative,
it is resolved against a provided base directory.

Arguments:
path_text (str): The input path string to resolve.
base_dir (Path): The base directory to resolve against if the input path is relative.

Returns:
Path: A Path object representing the resolved absolute path.
'''
def resolve_path(path_text: str, base_dir: Path) -> Path:
	raw = normalize_path_text(path_text)
	if raw.is_absolute():
		return raw
	return (base_dir / raw).resolve()

'''
# ----------------------------------------------------------------------------------------------------
# posix_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function converts a Path object to its POSIX-style string representation, which uses forward slashes
as separators. This is useful for ensuring consistent path formatting across different operating systems.

Arguments:
path (Path): The Path object to convert.

Returns:
str: The POSIX-style string representation of the path.
'''
def posix_path(path: Path) -> str:
	return path.resolve().as_posix()


'''
# ----------------------------------------------------------------------------------------------------
# normalize_split()
# ----------------------------------------------------------------------------------------------------
Description:
This function normalizes a split value (e.g., "train", "valid", "test") by stripping whitespace, converting
to lowercase, and mapping common aliases to standard split names. 

Arguments:
value (str): The split value to normalize.

Returns:
str: The normalized split value.
'''
def normalize_split(value: str) -> str:
	text = str(value or "").strip().lower()
	aliases = {
		"train": "train",
		"training": "train",
		"valid": "valid",
		"val": "valid",
		"validation": "valid",
		"test": "test",
		"testing": "test",
		"unknown": "unknown",
	}
	return aliases.get(text, "")

'''
# ----------------------------------------------------------------------------------------------------
# parse_source_stem()
# ----------------------------------------------------------------------------------------------------
Description:
This function extracts the source stem from a given file name. If the file name matches the expected augmented
pattern, it returns the source part of the name. Otherwise, it returns the stem of the file name (without extension).

Arguments:
file_name (str): The file name from which to extract the source stem.

Returns:
str: The extracted source stem.

'''
def parse_source_stem(file_name: str) -> str:
	match = AUGMENTED_PATTERN.match(file_name)
	if match:
		return match.group("source")
	return Path(file_name).stem

'''
# ----------------------------------------------------------------------------------------------------
# parse_augmented_filename()
# ----------------------------------------------------------------------------------------------------
Description:
This function parses an augmented file name to extract components such as the source stem, augmentation
type, and seed value. It uses a regular expression to match the expected pattern of augmented file names.

Arguments:
file_name (str): The augmented file name to parse.

Returns:
Dict[str, object]: A dictionary containing the extracted components (source_stem, augmentation, seed).

'''
def parse_augmented_filename(file_name: str) -> Dict[str, object]:
	match = AUGMENTED_PATTERN.match(file_name)
	if not match:
		return {}
	seed_hex = match.group("seed")
	try:
		seed_value = int(seed_hex, 16)
	except ValueError:
		seed_value = ""
	return {
		"source_stem": match.group("source"),
		"augmentation": match.group("aug"),
		"seed": seed_value,
	}

'''
# ----------------------------------------------------------------------------------------------------
# compute_similarity_key()
# ----------------------------------------------------------------------------------------------------
Description:
This function computes a similarity key for a given file name based on the specified similarity mode.
The similarity key is used to group files that are considered similar for the purpose of detecting
duplicates.

Arguments:
file_name (str): The file name for which to compute the similarity key.
source_stem (str): The source stem of the file.
mode (str): The similarity mode ("filename" or "source-stem").

Returns:
str: The computed similarity key.

'''
def compute_similarity_key(file_name: str, source_stem: str, mode: str) -> str:
	if mode == "filename":
		return file_name.lower()
	if mode == "source-stem":
		return source_stem.lower()
	normalized = SPLIT_TOKEN_PATTERN.sub("_", source_stem)
	return normalized.lower()


'''
# ----------------------------------------------------------------------------------------------------
# read_metadata_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function reads metadata rows from a CSV file at the specified path. It returns a list of dictionaries
representing the rows and a list of field names (column headers) from the CSV.

Arguments:
metadata_path (Path): The path to the metadata CSV file.

Returns:
Tuple[List[Dict[str, str]], List[str]]: A tuple containing the list of metadata rows and the list of field names.

'''
def read_metadata_rows(metadata_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
	if not metadata_path.exists():
		logging.warning("Metadata not found: %s", metadata_path)
		return [], []

	with open(metadata_path, "r", newline="", encoding="utf-8-sig") as handle:
		reader = csv.DictReader(handle)
		if not reader.fieldnames:
			logging.warning("Metadata CSV has no header: %s", metadata_path)
			return [], []
		return list(reader), list(reader.fieldnames)

'''
# ----------------------------------------------------------------------------------------------------
# build_fieldnames()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a list of field names for the output metadata CSV. It takes the existing field names
from the input CSV and ensures that all required output columns are included. If the input CSV has no header,
it returns the default output columns.

Arguments:
existing_fieldnames (List[str]): The list of existing field names from the input CSV.

Returns:
List[str]: The list of field names for the output metadata CSV.

'''
def build_fieldnames(existing_fieldnames: List[str]) -> List[str]:
	if not existing_fieldnames:
		return list(OUTPUT_COLUMNS)

	fieldnames = list(existing_fieldnames)
	for column in OUTPUT_COLUMNS:
		if column not in fieldnames:
			fieldnames.append(column)
	return fieldnames

'''
# ----------------------------------------------------------------------------------------------------
# make_backup()
# ----------------------------------------------------------------------------------------------------
Description:
This function creates a backup of the metadata CSV file by copying it to a new file with a timestamped
extension. If the original file does not exist, it returns None.

Arguments:
csv_path (Path): The path to the metadata CSV file to back up.

Returns:
Optional[Path]: The path to the backup file, or None if the original file does not exist.
'''
def make_backup(csv_path: Path) -> Optional[Path]:
	if not csv_path.exists():
		return None

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	backup_path = csv_path.parent / f"{csv_path.name}.bak_{timestamp}"
	shutil.copy2(csv_path, backup_path)
	return backup_path

'''
# ----------------------------------------------------------------------------------------------------
# write_metadata_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function writes metadata rows to a CSV file at the specified path. It takes a list of dictionaries
representing the rows, a list of field names for the CSV header, and an option to create a backup of the
existing metadata file before writing. It returns a list of backup file paths that were created.

Arguments:
metadata_path (Path): The path to the metadata CSV file.
fieldnames (List[str]): The list of field names for the CSV header.
rows (List[Dict[str, str]]): The list of metadata rows to write.
backup_enabled (bool): Whether to create a backup of the existing metadata file.

Returns:
List[Path]: A list of paths to the backup files that were created.

'''
def write_metadata_rows(
	metadata_path: Path,
	fieldnames: List[str],
	rows: List[Dict[str, str]],
	backup_enabled: bool,
) -> List[Path]:
	backups = []
	if backup_enabled:
		backup_path = make_backup(metadata_path)
		if backup_path:
			backups.append(backup_path)

	with open(metadata_path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({column: row.get(column, "") for column in fieldnames})

	return backups

'''
# ----------------------------------------------------------------------------------------------------
# filename_from_metadata()
# ----------------------------------------------------------------------------------------------------
Description:
This function extracts the filename from a given path text. It normalize the path text and returns the name
component of the path. If the input path text is empty, it returns an empty string.

Arguments:
path_text (str): The path text from which to extract the filename.

Returns:
str: The extracted filename or an empty string if the input is empty.

'''
def filename_from_metadata(path_text: str) -> str:
	if not path_text:
		return ""
	return normalize_path_text(path_text).name

'''
# ----------------------------------------------------------------------------------------------------
# build_rows_by_filename()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds an index of metadata rows by filename extracted from the "augmented_path" field.
It returns a dictionary where the keys are the lowercase filenames and the values are lists of metadata
row that have that filename in their "augmented_path". This allows for easy lookup of metadata rows base
on the augmented file names when processing the files on disk.

Arguments:
rows (Iterable[Dict[str, str]]): An iterable of metadata rows, where each row is a dictionary of column values.

Returns:
Dict[str, List[Dict[str, str]]]: A dictionary mapping lowercase filenames to lists of metadata rows.

'''
def build_rows_by_filename(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
	index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		name = filename_from_metadata(row.get("augmented_path", ""))
		if name:
			index[name.lower()].append(row)
	return index

'''
# ----------------------------------------------------------------------------------------------------
# build_metadata_split_index()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds an index of the most common split value for each filename based on the metadata rows.
It returns a dictionary where the keys are the lowercase filenames and the values are the most common split
associated with that filename in the metadata. This can be used to prefer the metadata split when resolving
duplicates on disk. If a filename has no valid split in the metadata, it will not be included in the index.

Arguments:
rows (Iterable[Dict[str, str]]): An iterable of metadata rows, where each row is a dictionary of column values.

Returns:
Dict[str, str]: A dictionary mapping lowercase filenames to their most common split values.

'''
def build_metadata_split_index(rows: Iterable[Dict[str, str]]) -> Dict[str, str]:
	split_counts: Dict[str, Counter] = defaultdict(Counter)
	for row in rows:
		name = filename_from_metadata(row.get("augmented_path", ""))
		split = normalize_split(row.get("split", ""))
		if name and split in SPLITS:
			split_counts[name.lower()][split] += 1

	preferred = {}
	for name, counts in split_counts.items():
		preferred[name] = counts.most_common(1)[0][0]
	return preferred


'''
# ----------------------------------------------------------------------------------------------------
# scan_image_files()
# ----------------------------------------------------------------------------------------------------
Description:
This function scans the specified images directory for image files in the known splits (train, valid, test, unknown).
It creates a list of FileRecord objects for each valid image file found, which includes the file path, split,
file name, similarity key (based on the specified similarity mode), and source stem. The function returns a list
of FileRecord objects representing the image files on disk.

Arguments:
images_dir (Path): The directory containing the image files.
similarity_mode (str): The mode for computing the similarity key.

Returns:
List[FileRecord]: A list of FileRecord objects representing the image files on disk.

'''
def scan_image_files(images_dir: Path, similarity_mode: str) -> List[FileRecord]:
	records: List[FileRecord] = []

	for split in KNOWN_SPLITS:
		split_dir = images_dir / split
		if not split_dir.exists():
			continue
		if not split_dir.is_dir():
			logging.warning("Images split path is not a directory: %s", split_dir)
			continue

		for file_path in sorted(split_dir.iterdir(), key=lambda path: path.name.lower()):
			if not file_path.is_file():
				continue
			if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
				continue

			source_stem = parse_source_stem(file_path.name)
			similarity_key = compute_similarity_key(file_path.name, source_stem, similarity_mode)
			records.append(
				FileRecord(
					path=file_path,
					split=split,
					name=file_path.name,
					similarity_key=similarity_key,
					source_stem=source_stem,
				)
			)

	return records

'''
# ----------------------------------------------------------------------------------------------------
# find_duplicate_groups()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes an iterable of FileRecord objects and groups them by their similarity key. It 
identifies groups of files that have the same similarity key but belong to different splits 
(train, valid, test). Only groups that contain files from more than one split are considered duplicates. 
The function returns a dictionary where the keys are the similarity keys and the values are lists of 
FileRecord objects that belong to the same duplicate group.

Arguments:
records (Iterable[FileRecord]): An iterable of FileRecord objects to analyze for duplicates.

Returns:
Dict[str, List[FileRecord]]: A dictionary where the keys are the similarity keys and the values 
                             are lists of FileRecord objects that belong to the same duplicate group.
							 
'''
def find_duplicate_groups(records: Iterable[FileRecord]) -> Dict[str, List[FileRecord]]:
	grouped: Dict[str, List[FileRecord]] = defaultdict(list)
	for record in records:
		grouped[record.similarity_key].append(record)

	duplicates = {}
	for key, group in grouped.items():
		splits = {item.split for item in group if item.split in SPLITS}
		if len(splits) > 1:
			duplicates[key] = group
	return duplicates

'''
# ----------------------------------------------------------------------------------------------------
# choose_preferred_split()
# ----------------------------------------------------------------------------------------------------
Description:
This function determines the preferred split for a group of similar files based on the specified split 
preference order and an optional preference for the metadata split. If prefer_metadata_split is True,
it first checks if any file in the group has a split in the metadata that matches the preferred splits.
If not, it falls back to checking the available splits in the group against the preferred order. 
It returns the chosen preferred split as a string.

Arguments:
group (List[FileRecord]): A list of FileRecord objects that belong to the same duplicate group.
split_preference (List[str]): A list of split names in order of preference (e.g., ["train", "valid", "test"]).
prefer_metadata_split (bool): Whether to prefer the split from metadata when determining the preferred split.
metadata_split_by_filename (Dict[str, str]): A dictionary mapping lowercase filenames to their most common split from metadata.

Returns:
str: The chosen preferred split as a string.

'''
def choose_preferred_split(
	group: List[FileRecord],
	split_preference: List[str],
	prefer_metadata_split: bool,
	metadata_split_by_filename: Dict[str, str],
) -> str:
	if prefer_metadata_split:
		for split in split_preference:
			for record in group:
				if metadata_split_by_filename.get(record.name.lower()) == split:
					return split

	available = {record.split for record in group if record.split in SPLITS}
	for split in split_preference:
		if split in available:
			return split
	return next(iter(available)) if available else ""

'''
# ----------------------------------------------------------------------------------------------------
# resolve_collision()
# ----------------------------------------------------------------------------------------------------
Description:
This function resolves filename collisions when moving files to the unknown directory. If the target path
already exists, it appends a suffix to the filename (e.g., __dup001) and checks for existence again. It continues
incrementing the suffix until it finds a filename that does not exist. If it cannot resolve the collision 
after a reasonable number of attempts, it raises an error.

Arguments:
target_path (Path): The target path for the file.

Returns:
Path: The resolved path for the file.

'''
def resolve_collision(target_path: Path) -> Path:
	if not target_path.exists():
		return target_path

	stem = target_path.stem
	suffix = target_path.suffix
	for idx in range(1, 1000):
		candidate = target_path.with_name(f"{stem}__dup{idx:03d}{suffix}")
		if not candidate.exists():
			return candidate

	raise RuntimeError(f"Unable to resolve filename collision: {target_path}")

'''
# ----------------------------------------------------------------------------------------------------
# move_file()
# ----------------------------------------------------------------------------------------------------
Description:
This function moves a file from a source path to a target directory. It first ensures that the target directory
exists, then it resolves any filename collisions in the target directory by using the resolve_collision function.
If dry_run is True, it does not actually move the file but returns the target path that would be used. If dry_run
is False, it performs the move operation and returns the final target path.

Arguments:
src_path (Path): The source path of the file to move.
target_dir (Path): The target directory to move the file into.
dry_run (bool): Whether to perform a dry run (do not actually move the file).

Returns:
Path: The final target path of the file after moving (or the intended target path if dry_run is True).

'''
def move_file(src_path: Path, target_dir: Path, dry_run: bool) -> Path:
	target_dir.mkdir(parents=True, exist_ok=True)
	target_path = resolve_collision(target_dir / src_path.name)
	if not dry_run:
		shutil.move(str(src_path), str(target_path))
	return target_path

'''
# ----------------------------------------------------------------------------------------------------
# apply_duplicate_moves()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes the list of FileRecord objects, identifies duplicate groups based on their similarity keys,
and moves the non-preferred files in each duplicate group to the unknown directory. It uses the choose_preferred_split
function to determine which split to prefer for each group. It keeps track of the reasons for moving files and the number
of duplicates and moved files. It returns a dictionary mapping lowercase filenames to the reason they were moved,
as well as the count of duplicate groups and moved files.

Arguments:
records (List[FileRecord]): A list of FileRecord objects representing the image files on disk.
unknown_dir (Path): The directory to move duplicate files into.
split_preference (List[str]): A list of split names in order of preference for resolving duplicates.
prefer_metadata_split (bool): Whether to prefer the split from metadata when determining the preferred split.
metadata_split_by_filename (Dict[str, str]): A dictionary mapping lowercase filenames to their most common split from metadata.
dry_run (bool): Whether to perform a dry run (do not actually move files).

Returns:
Tuple[Dict[str, str], int, int]: A tuple containing a dictionary mapping lowercase filenames to the reason they were moved,
                                  the count of duplicate groups, and the count of moved files.

'''
def apply_duplicate_moves(
	records: List[FileRecord],
	unknown_dir: Path,
	split_preference: List[str],
	prefer_metadata_split: bool,
	metadata_split_by_filename: Dict[str, str],
	dry_run: bool,
) -> Tuple[Dict[str, str], int, int]:
	duplicate_groups = find_duplicate_groups(records)
	moved_reason_by_name: Dict[str, str] = {}
	moved_files = 0

	if duplicate_groups:
		logging.info("Duplicate groups detected: %d", len(duplicate_groups))

	for key, group in duplicate_groups.items():
		preferred_split = choose_preferred_split(
			group,
			split_preference,
			prefer_metadata_split,
			metadata_split_by_filename,
		)
		logging.debug("Duplicate key %s preferred split: %s", key, preferred_split)

		for record in group:
			if record.split not in SPLITS:
				continue
			if record.split == preferred_split:
				continue

			target_path = move_file(record.path, unknown_dir, dry_run)
			moved_reason_by_name[target_path.name.lower()] = (
				f"duplicate: preferred split {preferred_split}"
			)
			moved_files += 1

	return moved_reason_by_name, len(duplicate_groups), moved_files

'''
# ----------------------------------------------------------------------------------------------------
# build_file_map()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a mapping of filenames to FileRecord objects from a list of FileRecord objects. It returns
a dictionary where the keys are the lowercase filenames and the values are the corresponding FileRecord objects. If there are
multiple files with the same name (case-insensitive), it keeps the first one and records the duplicates in a list.

Arguments:
records (Iterable[FileRecord]): An iterable of FileRecord objects to build the file map from.

Returns:
Tuple[Dict[str, FileRecord], List[str]]: A tuple containing a dictionary mapping lowercase filenames to FileRecord objects,
and a list of duplicate filenames that were encountered.

'''
def build_file_map(records: Iterable[FileRecord]) -> Tuple[Dict[str, FileRecord], List[str]]:
	file_map: Dict[str, FileRecord] = {}
	duplicates = []
	for record in records:
		key = record.name.lower()
		if key in file_map:
			duplicates.append(record.name)
			continue
		file_map[key] = record
	return file_map, duplicates

'''
# ----------------------------------------------------------------------------------------------------
# build_orphan_row()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a metadata row for a file that is present on disk but does not have a corresponding 
row in the metadata CSV. It parses the augmented filename to extract information such as the augmentation 
type and seed value, and it fills in the metadata fields accordingly. The status of the row is set to "ORPHAN"

Arguments:
record (FileRecord): The file record for which to build an orphan row.
fieldnames (List[str]): The list of field names for the metadata row.

Returns:
Dict[str, str]: The constructed metadata row for the orphan file.

'''
def build_orphan_row(record: FileRecord, fieldnames: List[str]) -> Dict[str, str]:
	parsed = parse_augmented_filename(record.name)
	row = {column: "" for column in fieldnames}
	row["source_path"] = ""
	row["augmented_path"] = posix_path(record.path)
	row["label"] = ""
	row["split"] = record.split
	row["augmentation"] = str(parsed.get("augmentation", ""))
	row["params_json"] = "{}"
	seed_value = parsed.get("seed", "")
	row["seed"] = str(seed_value) if seed_value != "" else ""
	row["status"] = "ORPHAN"
	row["error"] = "File present on disk without metadata row"
	return row

'''
# ----------------------------------------------------------------------------------------------------
# update_metadata_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function updates the metadata rows based on the current files on disk and the reasons for any moved 
files. It iterates through the existing metadata rows, checks if the augmented file exists on disk, and 
updates the "augmented_path", "split", "status", and "error" fields as needed. It also keeps track of 
statistics such as how many rows were updated, how many are missing, and how many orphan rows are added. 
If add_orphans is True, it also adds new metadata rows for any files that are present on disk but not in 
the metadata.

Arguments:
rows (List[Dict[str, str]]): The list of existing metadata rows to update.
file_by_name (Dict[str, FileRecord]): A dictionary mapping lowercase filenames to FileRecord objects representing the files on disk.
moved_reason_by_name (Dict[str, str]): A dictionary mapping lowercase filenames to the reason
                                        they were moved to the unknown directory.
add_orphans (bool): Whether to add metadata rows for files that are present on disk but not in the metadata.
fieldnames (List[str]): The list of field names for the metadata rows.

Returns:
Tuple[List[Dict[str, str]], Dict[str, int]]: A tuple containing the list of updated metadata rows and a dictionary of statistics about the updates.

'''
def update_metadata_rows(
	rows: List[Dict[str, str]],
	file_by_name: Dict[str, FileRecord],
	moved_reason_by_name: Dict[str, str],
	add_orphans: bool,
	fieldnames: List[str],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
	updated_rows: List[Dict[str, str]] = []
	seen_filenames = set()
	stats = {
		"rows_updated": 0,
		"rows_missing": 0,
		"orphans_added": 0,
	}

	for row in rows:
		updated = dict(row)
		name = filename_from_metadata(row.get("augmented_path", ""))

		if name:
			key = name.lower()
			seen_filenames.add(key)
			record = file_by_name.get(key)
			if record:
				new_path = posix_path(record.path)
				if updated.get("augmented_path", "") != new_path:
					updated["augmented_path"] = new_path
					stats["rows_updated"] += 1
				if normalize_split(updated.get("split", "")) != record.split:
					updated["split"] = record.split
					stats["rows_updated"] += 1
				if key in moved_reason_by_name:
					updated["status"] = "DUPLICATE"
					updated["error"] = moved_reason_by_name[key]
			else:
				status = str(updated.get("status", "")).strip().upper()
				if status not in {"FAILED", "SKIPPED"}:
					updated["status"] = "MISSING"
					updated["error"] = "File not found on disk"
					stats["rows_missing"] += 1

		updated_rows.append(updated)

	if add_orphans:
		for key, record in file_by_name.items():
			if key in seen_filenames:
				continue
			updated_rows.append(build_orphan_row(record, fieldnames))
			stats["orphans_added"] += 1

	return updated_rows, stats

'''
# ----------------------------------------------------------------------------------------------------
# sort_metadata_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function sorts the metadata rows based on a defined order of splits and then by filename. 
The sorting key is determined by the split (using the SPLIT_ORDER mapping), the source path, the 
augmented path, the augmentation type, and the seed value. This ensures that the output metadata 
is organized in a consistent and logical manner.

Arguments:
rows (List[Dict[str, str]]): The list of metadata rows to sort.

Returns:
List[Dict[str, str]]: The sorted list of metadata rows.

'''
def sort_metadata_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
	def sort_key(row: Dict[str, str]):
		split_key = SPLIT_ORDER.get(normalize_split(row.get("split", "")), 99)
		source_path = str(row.get("source_path", "")).lower()
		augmented_path = str(row.get("augmented_path", "")).lower()
		augmentation = str(row.get("augmentation", "")).lower()
		seed_text = str(row.get("seed", "")).strip()
		try:
			seed_value = int(seed_text)
		except ValueError:
			seed_value = 0
		return (split_key, source_path, augmented_path, augmentation, seed_value)

	return sorted(rows, key=sort_key)

'''
# ----------------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------------
Description:
This is the main function that orchestrates the process of checking for similar files, resolving duplicates,
and updating the metadata. It parses command-line arguments, sets up logging, resolves paths, reads metadata,
scans image files, applies duplicate moves, updates metadata rows, and writes the updated metadata back to disk.

Arguments:
None

Returns:
None

'''
def run() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
	images_dir = resolve_path(args.images_dir, output_dir)
	metadata_path = resolve_path(args.metadata, output_dir)
	unknown_dir = images_dir / "unknown"

	split_preference = [
		item.strip().lower() for item in args.split_preference.split(",") if item.strip()
	]
	split_preference = [item for item in split_preference if item in SPLITS]
	if not split_preference:
		split_preference = list(SPLITS)

	logging.info("Output directory: %s", output_dir)
	logging.info("Images directory: %s", images_dir)
	logging.info("Metadata path: %s", metadata_path)
	logging.info("Similarity mode: %s", args.similarity_mode)

	rows, existing_fieldnames = read_metadata_rows(metadata_path)
	fieldnames = build_fieldnames(existing_fieldnames)
	metadata_split_by_filename = build_metadata_split_index(rows)

	records = scan_image_files(images_dir, args.similarity_mode)
	logging.info("Images scanned: %d", len(records))

	moved_reason_by_name, duplicate_groups, moved_files = apply_duplicate_moves(
		records=records,
		unknown_dir=unknown_dir,
		split_preference=split_preference,
		prefer_metadata_split=args.prefer_metadata_split,
		metadata_split_by_filename=metadata_split_by_filename,
		dry_run=args.dry_run,
	)

	if args.dry_run:
		logging.info("Dry run: no files moved, metadata not written.")
		logging.info("Duplicate groups: %d | Files moved: %d", duplicate_groups, moved_files)
		return

	if moved_files:
		records = scan_image_files(images_dir, args.similarity_mode)

	file_by_name, duplicate_names = build_file_map(records)
	if duplicate_names:
		logging.warning("Duplicate filenames after moves: %s", duplicate_names[:5])

	updated_rows, update_stats = update_metadata_rows(
		rows=rows,
		file_by_name=file_by_name,
		moved_reason_by_name=moved_reason_by_name,
		add_orphans=args.add_orphans,
		fieldnames=fieldnames,
	)

	if args.sort_metadata:
		updated_rows = sort_metadata_rows(updated_rows)

	backups = write_metadata_rows(
		metadata_path=metadata_path,
		fieldnames=fieldnames,
		rows=updated_rows,
		backup_enabled=not args.no_backup,
	)

	logging.info("Duplicate groups: %d | Files moved: %d", duplicate_groups, moved_files)
	logging.info(
		"Metadata rows updated: %d | Missing marked: %d | Orphans added: %d",
		update_stats["rows_updated"],
		update_stats["rows_missing"],
		update_stats["orphans_added"],
	)
	if backups:
		logging.info("Metadata backup created: %s", backups[0])

# Main entry point
if __name__ == "__main__":
	run()