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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SPLITS = ("train", "valid", "test")
KNOWN_SPLITS = ("train", "valid", "test", "unknown")
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2, "unknown": 3}

OUTPUT_COLUMNS = [
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


# ----------------------------------------------------------------------------------------------------
# Argument parsing / logging
# ----------------------------------------------------------------------------------------------------
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


def setup_logging(level: str) -> None:
	logging.basicConfig(
		level=getattr(logging, level),
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


# ----------------------------------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------------------------------
def normalize_path_text(path_text: str) -> Path:
	return Path(path_text.replace("\\", "/"))


def resolve_path(path_text: str, base_dir: Path) -> Path:
	raw = normalize_path_text(path_text)
	if raw.is_absolute():
		return raw
	return (base_dir / raw).resolve()


def posix_path(path: Path) -> str:
	return path.resolve().as_posix()


# ----------------------------------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------------------------------
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


def parse_source_stem(file_name: str) -> str:
	match = AUGMENTED_PATTERN.match(file_name)
	if match:
		return match.group("source")
	return Path(file_name).stem


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


def compute_similarity_key(file_name: str, source_stem: str, mode: str) -> str:
	if mode == "filename":
		return file_name.lower()
	if mode == "source-stem":
		return source_stem.lower()
	normalized = SPLIT_TOKEN_PATTERN.sub("_", source_stem)
	return normalized.lower()


# ----------------------------------------------------------------------------------------------------
# Metadata I/O
# ----------------------------------------------------------------------------------------------------
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


def build_fieldnames(existing_fieldnames: List[str]) -> List[str]:
	if not existing_fieldnames:
		return list(OUTPUT_COLUMNS)

	fieldnames = list(existing_fieldnames)
	for column in OUTPUT_COLUMNS:
		if column not in fieldnames:
			fieldnames.append(column)
	return fieldnames


def make_backup(csv_path: Path) -> Optional[Path]:
	if not csv_path.exists():
		return None

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	backup_path = csv_path.parent / f"{csv_path.name}.bak_{timestamp}"
	shutil.copy2(csv_path, backup_path)
	return backup_path


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


def filename_from_metadata(path_text: str) -> str:
	if not path_text:
		return ""
	return normalize_path_text(path_text).name


def build_rows_by_filename(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
	index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
	for row in rows:
		name = filename_from_metadata(row.get("augmented_path", ""))
		if name:
			index[name.lower()].append(row)
	return index


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


# ----------------------------------------------------------------------------------------------------
# File scanning and moves
# ----------------------------------------------------------------------------------------------------
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


def move_file(src_path: Path, target_dir: Path, dry_run: bool) -> Path:
	target_dir.mkdir(parents=True, exist_ok=True)
	target_path = resolve_collision(target_dir / src_path.name)
	if not dry_run:
		shutil.move(str(src_path), str(target_path))
	return target_path


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


# ----------------------------------------------------------------------------------------------------
# Metadata updates
# ----------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------------------------------------
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


if __name__ == "__main__":
	run()