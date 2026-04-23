"""
Update the metadata of the dataset.

Author: Axewhy
Date: 2026-04-23
Last Updated: 2026-04-27

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
SPLITS                                              = ("train", "valid", "test")
SPLIT_ORDER                                         = {"train": 0, "valid": 1, "test": 2}
SUPPORTED_IMAGE_EXTENSIONS                          = {".jpg", ".jpeg", ".png", ".webp"}

STREET_COLUMNS                                      = (
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

MAP_COLUMNS                                         = (
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

'''
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
Description:
This function parses command-line arguments for the metadata reconciliation script. It supports options
for selecting the dataset to update (street, map, or both), specifying the base directory containing the
Dataset folder, enabling a dry run mode that performs checks without writing updates, dropping non-file
rows from the output, disabling backup creation before writing updates, and setting the logging level.

Arguments:
--dataset: Choice of dataset to update (street, map, or both). Default is "both".
--base-dir: Base directory containing the Dataset folder. Default is the current directory.
--dry-run: If set, the script will run checks and provide a summary without writing any CSV files.
--drop-nonfile-rows: If set, rows with empty paths (non-file rows) will be dropped from the output.
--no-backup: If set, the script will not create backup CSV files before writing updates.
--log-level: Logging level for the script output (DEBUG, INFO, WARNING, ERROR). Default is "INFO".

Returns:
An argparse.Namespace object containing the parsed command-line arguments.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
Description:
This function sets up the logging configuration for the script. It configures the logging level, format,
and date format for log messages. The logging level is determined by the log_level argument, which can be
set to DEBUG, INFO, WARNING, or ERROR. The log messages will include a timestamp, the log level, and the
message itself.

Arguments:
log_level: A string representing the logging level (DEBUG, INFO, WARNING, ERROR).

Returns:
None
'''
def setup_logging(log_level):
	logging.basicConfig(
		level=getattr(logging, log_level),
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

'''
# ----------------------------------------------------------------------------------------------------
# string_value()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes an input value and returns a stripped string representation of it. If the input
value is None, it returns an empty string. This function is used to ensure that all values are consistently
treated as strings and that any leading or trailing whitespace is removed.

Arguments:
value: The input value to be converted to a string.

Returns:
A string representation of the input value, with leading and trailing whitespace removed. If the input
value is None, it returns an empty string.
'''
def string_value(value):
	if value is None:
		return ""
	return str(value).strip()

'''
# ----------------------------------------------------------------------------------------------------
# normalize_split()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes an input value and normalizes it to a valid split name (train, valid, or test). It
converts the input value to a lowercase string and checks if it is one of the valid split names. If it is
valid, it returns the normalized split name; otherwise, it returns an empty string. This function is used
to ensure that split names are consistently formatted and valid throughout the script.

Arguments:
value: The input value to be normalized as a split name.

Returns:
A normalized split name (train, valid, or test) if the input value is valid; otherwise, an empty string.
'''
def normalize_split(value):
	text = string_value(value).lower()
	return text if text in SPLITS else ""

'''
# ----------------------------------------------------------------------------------------------------
# normalize_int_text()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes an input value and attempts to convert it to an integer. If the conversion is
successful, it returns the integer as a string. Otherwise, it returns the original string value.
This function is used to normalize values that are expected to be integers (such as heading, pitch, fov)
while preserving non-integer values when the conversion fails.

Arguments:
value: The input value to be normalized as an integer text.

Returns:
A string representation of the integer value if the conversion is successful;
otherwise, the original string value.

Raises:
ValueError: If the input value cannot be converted to an integer or float when it is expected to be numeric.
'''
def normalize_int_text(value):
	text = string_value(value)
	if not text:
		return ""
	try:
		return str(int(float(text)))
	except ValueError:
		return text

'''
# ----------------------------------------------------------------------------------------------------
# numeric_sort_key()
# ----------------------------------------------------------------------------------------------------
Description:
This function generates a sort key for a given value that can be used for numeric-aware sorting. It
first attempts to convert the input value to an integer. If that fails, it tries to convert it to a float
and then to an integer. If both conversions fail, it treats the value as a string. The sort key is a
tuple where the first element indicates the type (0 for numeric, 1 for non-numeric, 2 for empty), the second
element is the numeric value (or 0 for non-numeric), and the third element is the original string value.
This allows for sorting that prioritizes numeric values while still handling non-numeric and empty values
consistently.

Arguments:
value: The input value for which to generate a numeric sort key.

Returns:
A tuple representing the sort key, where:
- The first element is an integer indicating the type (0 for numeric, 1 for non-numeric, 2 for empty).
- The second element is the numeric value if the input is numeric, or 0 for non-numeric values.
- The third element is the original string value for non-numeric inputs, or an empty string for numeric inputs.

Raises:
ValueError: If the input value cannot be converted to an integer or float when it is expected to be numeric.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# row_score()
# ----------------------------------------------------------------------------------------------------
Description:
This function calculates a score for a given row based on the presence of values in specified columns.
It iterates through the provided columns and counts how many of them have non-empty string values in the
row. The score is the total count of columns that contain non-empty values. This function is used to
evaluate the completeness of a row, with higher scores indicating that more columns have values, which can 
be useful when selecting between muliple rows for the same file key to determine which one has more metadata
information to fill in missing details for a file.

Arguments:
row: A dictionary representing a row of metadata, where keys are column names 
     and values are the corresponding cell values.
columns: An iterable of column names to check for non-empty values in the row.

Returns:
int: The score representing the number of columns with non-empty values in the row.
'''
def row_score(row, columns):
	return sum(1 for column in columns if string_value(row.get(column)))

'''
# ----------------------------------------------------------------------------------------------------
# make_backup()
# ----------------------------------------------------------------------------------------------------
Description:
This function creates a backup of the specified CSV file by copying it to a new file with a timestamped
extension. If the specified CSV file does not exist, the function returns None and does not create a backup.
The backup file is created in the same directory as the original CSV file, and its name is formed by appending
".bak_" followed by the current timestamp (in the format YYYYMMDD_HHMMSS) to the original file name. 
The function returns the path to the backup file if it was created successfully.

Arguments:
csv_path: A Path object representing the path to the CSV file that should be backed up.

Returns:
A Path object representing the path to the backup file if the backup was created successfully; 
otherwise, None if the original CSV file does not exist.
'''
def make_backup(csv_path):
	if not csv_path.exists():
		return None

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	backup_path = csv_path.parent / f"{csv_path.name}.bak_{timestamp}"
	shutil.copy2(csv_path, backup_path)
	return backup_path

'''
# ----------------------------------------------------------------------------------------------------
# read_csv_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function reads rows from a CSV file and returns them as a list of dictionaries. Each dictionary
represents a row in the CSV, with keys corresponding to the column headers and values corresponding to
the cell values. If the specified CSV file does not exist, the function logs an informational message and
returns an empty list. If the CSV file exists but has no header, the function logs a warning and also returns
an empty list. The function uses UTF-8 encoding with a signature (BOM) to ensure proper handling of 
CSV files that may contain special characters.

Arguments:
csv_path: A Path object representing the path to the CSV file to be read.

Returns:
A list of dictionaries, where each dictionary represents a row from the CSV file. The keys of the dictionary
are the column headers from the CSV, and the values are the corresponding cell values for that row. If the CSV
file does not exist or has no header, the function returns an empty list.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# normalize_row()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a row represented as a dictionary and normalizes its values based on the specified columns.
It creates a new dictionary where each key is a column name from the provided columns, and the corresponding
value is the string representation of the value from the input row for that column. If a column is missing
in the input row, it defaults to an empty string. This function ensures that all rows have a consistent set of
columns with string values, which can be important for later processing and comparison of rows.

Arguments:
row: A dictionary representing a row of metadata, where keys are column names 
	 and values are the corresponding cell values.
columns: An iterable of column names that should be present in the normalized row.

Returns:
A new dictionary where each key is a column name from the provided columns, and the value is the string
representation of the corresponding value from the input row. If a column is missing in the input row
or its value is None, the value in the normalized row will be an empty string.
'''
def normalize_row(row, columns):
	normalized = {column: "" for column in columns}
	for column in columns:
		normalized[column] = string_value(row.get(column, ""))
	return normalized

'''
# ----------------------------------------------------------------------------------------------------
# write_csv_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function writes a list of rows to a CSV file at the specified path. It first ensures that the parent
directory of the CSV file exists, creating it if necessary. If backup_enabled is True, it creates a backup
of the existing CSV file before writing the new rows. The function then opens the CSV file for writing with
UTF-8 encoding and uses a csv.DictWriter to write the header and rows to the file. Each row is written as a
dictionary where the keys are the column names and the values are the corresponding cell values. The function
also appends the path of any backup file created to the backups list provided as an argument.

Arguments:
csv_path: A Path object representing the path to the CSV file where the rows should be written.
columns: An iterable of column names that should be used as the header for the CSV file.
rows: A list of dictionaries, where each dictionary represents a row to be written to the CSV file. The keys
	  of the dictionary should correspond to the column names.
backup_enabled: A boolean indicating whether a backup of the existing CSV file should be created before writing.
backups: A list to which the path of any backup file created will be appended.

Returns:
None
'''
def write_csv_rows(csv_path, columns, rows, backup_enabled, backups):
	csv_path.parent.mkdir(parents=True, exist_ok=True)

	if backup_enabled:
		backup_path = make_backup(csv_path)
		if backup_path is not None:
			backups.append(str(backup_path))

	with open(csv_path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(columns))
		writer.writeheader()
		for row in rows:
			writer.writerow({column: row.get(column, "") for column in columns})

'''
# ----------------------------------------------------------------------------------------------------
# parse_street_filename()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a file name as input and attempts to parse it according to the expected format for
street dataset images. The expected format is: {id}_{split}_h{heading}_p{pitch}_f{fov}.{ext}, where:
- {id} is a unique identifier for the sample.
- {split} is the dataset split (train, valid, or test).
- {heading} is the heading angle of the street view image.
- {pitch} is the pitch angle of the street view image.
- {fov} is the field of view of the street view image.
- {ext} is the file extension (e.g., jpg, png).
The function uses a regular expression to match the file name against this pattern. If the file name matches,
it extracts the components and returns them in a dictionary. If the file name does not match the expected format,
the function returns None.

Arguments:
file_name: A string representing the name of the file to be parsed.

Returns:
A dictionary containing the parsed components of the file name if it matches the expected format, with keys:
- "id": The unique identifier for the sample.
- "split": The dataset split (train, valid, or test).
- "heading": The heading angle of the street view image.
- "pitch": The pitch angle of the street view image.
- "fov": The field of view of the street view image.
- "ext": The file extension (e.g., jpg, png).
If the file name does not match the expected format, the function returns None.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# parse_map_filename()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a file name as input and attempts to parse it according to the expected format for
map dataset images. The expected format is: {id}_{split}_z{zoom}_x{tile_x}_y{tile_y}.{ext}, where:
- {id} is a unique identifier for the sample.
- {split} is the dataset split (train, valid, or test).
- {zoom} is the zoom level of the map tile.
- {tile_x} is the x coordinate of the map tile.
- {tile_y} is the y coordinate of the map tile.
- {ext} is the file extension (e.g., jpg, png).
The function uses a regular expression to match the file name against this pattern. If the file name matches,
it extracts the components and returns them in a dictionary. If the file name does not match the expected format,
the function returns None.

Arguments:
file_name: A string representing the name of the file to be parsed.

Returns:
A dictionary containing the parsed components of the file name if it matches the expected format, with keys:
- "id": The unique identifier for the sample.
- "split": The dataset split (train, valid, or test).
- "zoom": The zoom level of the map tile.
- "tile_x": The x coordinate of the map tile.
- "tile_y": The y coordinate of the map tile.
- "ext": The file extension (e.g., jpg, png).
If the file name does not match the expected format, the function returns None.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# parse_filename()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a dataset name and a file name as input and attempts to parse the file name according
to the expected format for the specified dataset. It supports two dataset types, "street" and "map", each
with its own expected file name format. The function delegates the parsing to the appropriate parsing function
based on the dataset name. If the dataset name is "street", it calls parse_street_filename. If the dataset name
is "map", it calls parse_map_filename. If the dataset name is not recognized, the function returns None.

Arguments:
dataset_name: A string representing the name of the dataset ("street" or "map") for 
			  which the file name should be parsed.
file_name: A string representing the name of the file to be parsed.

Returns:
A dictionary containing the parsed components of the file name if it matches the expected format for the
specified dataset. The keys in the dictionary will depend on the dataset type:
- For "street" dataset:
  - "id": The unique identifier for the sample.
  - "split": The dataset split (train, valid, or test).
  - "heading": The heading angle of the street view image.
  - "pitch": The pitch angle of the street view image.
  - "fov": The field of view of the street view image.
  - "ext": The file extension (e.g., jpg, png).
- For "map" dataset:
  - "id": The unique identifier for the sample.
  - "split": The dataset split (train, valid, or test).
  - "zoom": The zoom level of the map tile.
  - "tile_x": The x coordinate of the map tile.
  - "tile_y": The y coordinate of the map tile.
  - "ext": The file extension (e.g., jpg, png).
If the file name does not match the expected format for the specified dataset, or if the 
dataset name is not recognized, the function returns None.
'''
def parse_filename(dataset_name, file_name):
	if dataset_name == "street":
		return parse_street_filename(file_name)
	if dataset_name == "map":
		return parse_map_filename(file_name)
	return None

'''
# ----------------------------------------------------------------------------------------------------
# build_key_from_parts()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a dataset name and a dictionary of parts extracted from a file name and builds a
key tuple that uniquely identifies a file in the dataset. The key is constructed based on the dataset
type. For the "street" dataset, the key consists of the sample ID, split name, heading, pitch, and
field of view. For the "map" dataset, the key consists of the sample ID, split name, zoom level, tile_x,
and tile_y. The function first normalizes the split name and sample ID, and then checks if the necessary
components are present. If any required component is missing or invalid, the function returns None. Otherwise,
it returns the constructed key tuple.

Arguments:
dataset_name: A string representing the name of the dataset ("street" or "map") for which the key should be built.
parts: A dictionary containing the components extracted from a file name, which may include:
- "id": The unique identifier for the sample.
- "split": The dataset split (train, valid, or test).
- For "street" dataset:
  - "heading": The heading angle of the street view image.
  - "pitch": The pitch angle of the street view image.
  - "fov": The field of view of the street view image.

- For "map" dataset:
  - "zoom": The zoom level of the map tile.
  - "tile_x": The x coordinate of the map tile.
  - "tile_y": The y coordinate of the map tile.

Returns:
A tuple representing the key that uniquely identifies a file in the dataset if all required components are valid;
otherwise, None if any required component is missing or invalid.
'''
def build_key_from_parts(dataset_name, parts):
	split_name = normalize_split(parts.get("split", ""))
	sample_id = string_value(parts.get("id", ""))

	if not sample_id or not split_name:
		return None

	if dataset_name == "street":
		heading = normalize_int_text(parts.get("heading", ""))
		pitch = normalize_int_text(parts.get("pitch", ""))
		fov = normalize_int_text(parts.get("fov", ""))
		if not heading or not pitch or not fov:
			return None
		return (sample_id, split_name, heading, pitch, fov)

	zoom = normalize_int_text(parts.get("zoom", ""))
	tile_x = normalize_int_text(parts.get("tile_x", ""))
	tile_y = normalize_int_text(parts.get("tile_y", ""))
	if not zoom or not tile_x or not tile_y:
		return None
	return (sample_id, split_name, zoom, tile_x, tile_y)

'''
# ----------------------------------------------------------------------------------------------------
# build_key_from_row()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a dataset name and a row of metadata (as a dictionary) and attempts to build a key
tuple that uniquely identifies a file in the dataset. It first extracts relevant parts from the row,
such as the sample ID, split name, and other components depending on the dataset type. It then attempts
to build the key using the build_key_from_parts function. If the key cannot be built from the row parts,
it tries to parse the file name from the "path" field of the row to extract the necessary components to
build the key. If the file name can be parsed successfully, it uses the parsed components to build the key.
If all attempts to build the key fail, the function returns None. This function is used to determine the
unique identifier for a metadata row based on its content, which can then be used to match it against files
on disk.

Arguments:
dataset_name: A string representing the name of the dataset ("street" or "map") for which the key should be built.
row: A dictionary representing a row of metadata, where keys are column names and values are 
	 the corresponding cell values.

Returns:
A tuple representing the key that uniquely identifies a file in the dataset if it can be built from the row;
otherwise, None if the key cannot be built from the row parts or the file name cannot be parsed successfully.
'''
def build_key_from_row(dataset_name, row):
	parts = {
		"id": string_value(row.get("id", "")),
		"split": normalize_split(row.get("split", "")),
	}

	if dataset_name == "street":
		parts["heading"] = normalize_int_text(row.get("heading", ""))
		parts["pitch"] = normalize_int_text(row.get("pitch", ""))
		parts["fov"] = normalize_int_text(row.get("fov", ""))
	else:
		parts["zoom"] = normalize_int_text(row.get("zoom", ""))
		parts["tile_x"] = normalize_int_text(row.get("tile_x", ""))
		parts["tile_y"] = normalize_int_text(row.get("tile_y", ""))

	key = build_key_from_parts(dataset_name, parts)
	if key is not None:
		return key

	path_text = string_value(row.get("path", ""))
	if not path_text:
		return None

	file_name = Path(path_text.replace("\\", "/")).name
	parsed = parse_filename(dataset_name, file_name)
	if parsed is None:
		return None

	if parts["split"]:
		parsed["split"] = parts["split"]

	if parts["id"]:
		parsed["id"] = parts["id"]

	return build_key_from_parts(dataset_name, parsed)

'''
# ----------------------------------------------------------------------------------------------------
# image_format_from_extension()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a file extension as input and returns the corresponding image format string. It first
normalizes the extension by converting it to lowercase and stripping any leading periods. If the extension
is "jpg", it returns "jpeg" as the image format, since "jpeg" is the standard format name used in metadata.
For any other extension, it returns the normalized extension as the image format. This function is used to
determine the image format for map tiles based on their file extensions, ensuring that the metadata reflects
the correct format regardless of how the file is named on disk.

Arguments:
extension: A string representing the file extension (e.g., "jpg", "png") from which to derive the image format.

Returns:
A string representing the image format corresponding to the given file extension. If the extension is "jpg",
it returns "jpeg". For any other extension, it returns the normalized extension itself as the image format.
'''
def image_format_from_extension(extension):
	ext = string_value(extension).lower().lstrip(".")
	if ext == "jpg":
		return "jpeg"
	return ext

'''
# ----------------------------------------------------------------------------------------------------
# build_metadata_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function constructs the metadata path for a given file based on the dataset configuration, split
name, and file name. It uses the path prefix from the dataset configuration, the split directory name
corresponding to the split name, and the file name to build a Windows-style path that can be used in
the metadata CSV. The function first retrieves the split directory name from the dataset configuration
using the split name, then constructs the path by combining the path prefix, split directory name, and
file name using the PureWindowsPath class to ensure that the resulting path is in the correct format for
Windows. Finally, it returns the constructed path as a string. This function is used to generate the
correct path to the image files in the metadata based on the file's location on disk and the dataset
structure defined in the configuration.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, 
		including path prefix and split directories.
split_name: A string representing the name of the dataset split (train, valid, or test) 
			for which to build the metadata path.
file_name: A string representing the name of the file for which to build the metadata path.

Returns:
A string representing the constructed metadata path for the given file, formatted as a Windows-style path.
'''
def build_metadata_path(config, split_name, file_name):
	split_dir_name = Path(config.split_dirs[split_name]).name
	win_path = PureWindowsPath(config.path_prefix) / split_dir_name / file_name
	return str(win_path)

'''
# ------------------------------------------------------------------------------------
# scan_split_files()
# ------------------------------------------------------------------------------------
Description:
This function scans the directories for each dataset split (train, valid, test) as defined
in the dataset configuration and collects information about the image files found. It 
iterates through each split directory, checks for supported image files, and attempts to 
parse the file names to extract metadata components. The function builds a dictionary of files
indexed by a unique key derived from the parsed file name components. It also collects
statistics about the number of files scanned, any parsing errors encountered, and any duplicate
keys found. The resulting dictionary of files includes the parsed components, the file name,
the absolute path to the file, and the corresponding metadata path that should be used in the CSV.
The function returns a dictionary containing the files indexed by key, the total number of files
scanned, a list of parsing errors, and a list of duplicate keys with their corresponding file paths.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, 
		including split directories and path prefix.
base_dir: A Path object representing the base directory containing the split directories.

Returns:
A dictionary with the following structure:
{
	"files_by_key": A dictionary where keys are unique identifiers derived 
					from the parsed file name components, and values are dictionaries containing:
		- "parts": The parsed components extracted from the file name.
		- "file_name": The name of the file.
		- "abs_path": The absolute path to the file on disk.
		- "metadata_path": The path that should be used in the metadata CSV for this file.
	"files_scanned": The total number of files that were scanned across all split directories.
	"parse_errors": A list of file paths that could not be parsed successfully.
	"duplicate_keys": A list of tuples, where each tuple contains a duplicate key and the 
					  corresponding file paths that share that key.
}
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# select_best_row()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes two rows of metadata and a list of columns, and selects the row that has the highest
score based on the presence of non-empty values in the specified columns. The score for each row is 
calculated using the row_score function, which counts how many of the specified columns have non-empty
values. If the current_row is None, the function returns the candidate_row. If the candidate_row has 
a higher score than the current_row, it returns the candidate_row. Otherwise, it returns the current_row.
This function is used to determine which of two rows should be preferred when they represent the same file
key, with the goal of retaining the row that has more metadata information filled in for the specified columns.

Arguments:
current_row: A dictionary representing the current row of metadata, or None if there is no current row.
candidate_row: A dictionary representing the candidate row of metadata that is being considered for selection.
columns: An iterable of column names that should be considered when calculating the score for each row.

Returns:
A dictionary representing the row that has the highest score based on the presence of non-empty values 
in the specified columns. If current_row is None, it returns candidate_row. 
If candidate_row has a higher score than current_row, it returns candidate_row; 
otherwise, it returns current_row.
'''
def select_best_row(current_row, candidate_row, columns):
	if current_row is None:
		return candidate_row

	if row_score(candidate_row, columns) > row_score(current_row, columns):
		return candidate_row

	return current_row

'''
# ----------------------------------------------------------------------------------------------------
# build_context_by_key()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a context dictionary indexed by file keys from a list of existing metadata rows. It
iterates through the existing rows and attempts to build a key for each row using the build_key_from_row
function. If a key can be built successfully, it normalizes the row and uses the select_best_row function
to determine if this row should be stored in the context for that key, based on the presence of metadata
values in the specified columns. The function also counts how many rows had invalid file keys that could
not be built. The resulting context_by_key dictionary maps each valid file key to the best corresponding
metadata row found in the existing rows. The function returns the context_by_key dictionary and the count of
invalid file rows.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, including column names.
existing_rows: A list of dictionaries, where each dictionary represents an existing row of metadata. The keys
			   of the dictionary are column names, and the values are the corresponding cell values for that row.

Returns:
A tuple containing:
- context_by_key: A dictionary where keys are file keys built from the existing rows, 
				  and values are the best corresponding metadata rows (as dictionaries) 
				  for those keys based on the presence of non-empty values in the specified columns.
- invalid_file_rows: An integer count of how many existing rows had invalid file keys that could not be built.
'''
def build_context_by_key(config, existing_rows):
	context_by_key = {}
	invalid_file_rows = 0

	for row in existing_rows:
		if not string_value(row.get("path", "")):
			continue

		key = build_key_from_row(config.name, row)
		if key is None:
			invalid_file_rows += 1
			continue

		normalized = normalize_row(row, config.columns)
		context_by_key[key] = select_best_row(
			context_by_key.get(key),
			normalized,
			config.columns,
		)

	return context_by_key, invalid_file_rows

'''
# ----------------------------------------------------------------------------------------------------
# build_id_context()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a context dictionary indexed by sample IDs from a list of existing metadata rows. It
iterates through the existing rows and extracts the sample ID from each row. If a sample ID is present, it
normalizes the row and uses the select_best_row function to determine if this row should be stored in the
context for that sample ID, based on the presence of metadata values in the specified columns. The resulting
context_by_id dictionary maps each sample ID to the best corresponding metadata row found in the existing rows.
The function returns the context_by_id dictionary, which can be used to look up metadata rows based on sample
IDs, allowing for filling in missing metadata for files that share the same sample ID but may have different
file keys (e.g., due to differences in heading, pitch, fov for street dataset or zoom, tile_x, tile_y for map dataset).

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, including column names.
existing_rows: A list of dictionaries, where each dictionary represents an existing row of metadata. The keys
			   of the dictionary are column names, and the values are the corresponding cell values for that row.

Returns:
A dictionary where keys are sample IDs extracted from the existing rows, and values are the best corresponding
metadata rows (as dictionaries) for those sample IDs based on the presence of non-empty values in the specified columns.
'''
def build_id_context(config, existing_rows):
	context_by_id = {}

	for row in existing_rows:
		sample_id = string_value(row.get("id", ""))
		if not sample_id:
			continue

		normalized = normalize_row(row, config.columns)
		context_by_id[sample_id] = select_best_row(
			context_by_id.get(sample_id),
			normalized,
			config.columns,
		)

	return context_by_id

'''
# ----------------------------------------------------------------------------------------------------
# dedupe_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function takes a list of rows and a list of columns, and returns a new list of rows with duplicates 
removed based on the specified columns. It creates a set to keep track of seen signatures, where each 
signature is a tuple of string values for the specified columns in a row. As it iterates through the 
input rows, it generates the signature for each row and checks if it has been seen before. If the 
signature is new, it adds it to the seen set and includes the row in the unique_rows list. If the 
signature has already been seen, it skips that row. The resulting unique_rows list contains only one 
instance of each unique combination of values for the specified columns.

Arguments:
rows: A list of dictionaries, where each dictionary represents a row of metadata. The keys of the dictionary
	  are column names, and the values are the corresponding cell values for that row.
columns: An iterable of column names that should be considered when determining if rows are duplicates.

Returns:
A list of dictionaries representing the unique rows from the input list, where uniqueness is determined
based on the combination of string values for the specified columns. If multiple rows have the same values
for the specified columns, only the first occurrence will be included in the returned list.
'''
def dedupe_rows(rows, columns):
	seen = set()
	unique_rows = []

	for row in rows:
		signature = tuple(string_value(row.get(column, "")) for column in columns)
		if signature in seen:
			continue
		seen.add(signature)
		unique_rows.append(row)

	return unique_rows

'''
# ----------------------------------------------------------------------------------------------------
# build_row_for_file()
# ----------------------------------------------------------------------------------------------------
Description:
This function builds a metadata row for a given file based on the dataset configuration, file information,
and existing context rows. It takes the parsed components of the file name, the file name itself, and the
metadata path, and constructs a new row dictionary with the appropriate values for each column. It also
attempts to fill in missing metadata values from the existing context rows based on the file key and sample ID.
The function ensures that the required fields such as "id", "split", "path", and "status" are populated, 
and it fills in additional metadata fields based on the dataset type (street or map). The resulting row 
is a complete representation of the metadata for the given file, which can then be included in the output CSV.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, including column names and dataset type.
file_info: A dictionary containing information about the file, including:
- "parts": The parsed components extracted from the file name.
- "file_name": The name of the file.
- "abs_path": The absolute path to the file on disk.
- "metadata_path": The path that should be used in the metadata CSV for this file.
key_row: A dictionary representing the existing metadata row that matches the file key, or None if no match is found.
id_row: A dictionary representing the existing metadata row that matches the sample ID, or None if no match is found.

Returns:
A dictionary representing the metadata row for the given file, with values populated based on the file information
and existing context rows. The keys of the dictionary correspond to the column names defined in the dataset 
configuration, and the values are the corresponding cell values for that row.
'''
def build_row_for_file(config, file_info, key_row, id_row):
	parts = file_info["parts"]
	row = {column: "" for column in config.columns}

	for source in (id_row, key_row):
		if source is None:
			continue
		for column in config.columns:
			value = string_value(source.get(column, ""))
			if value:
				row[column] = value

	row["id"] = string_value(parts["id"])
	row["split"] = normalize_split(parts["split"])
	row["path"] = string_value(file_info["metadata_path"])
	row["status"] = "OK"
	row["error"] = ""

	if config.name == "street":
		row["heading"] = normalize_int_text(parts["heading"])
		row["pitch"] = normalize_int_text(parts["pitch"])
		row["fov"] = normalize_int_text(parts["fov"])
		if not row["size"]:
			row["size"] = "640x640"
	else:
		row["zoom"] = normalize_int_text(parts["zoom"])
		row["tile_x"] = normalize_int_text(parts["tile_x"])
		row["tile_y"] = normalize_int_text(parts["tile_y"])
		row["image_format"] = image_format_from_extension(parts["ext"])
		if not row["map_type"]:
			row["map_type"] = "satellite"
		if not row["tile_width"]:
			row["tile_width"] = "256"
		if not row["tile_height"]:
			row["tile_height"] = "256"

	return row

'''
# ----------------------------------------------------------------------------------------------------
# sort_rows()
# ----------------------------------------------------------------------------------------------------
Description:
This function sorts a list of metadata rows based on a defined order of splits and then by specific 
metadata fields. The sorting key is constructed as a tuple that first considers the split order 
(train, valid, test), then the sample ID, and finally additional fields depending on the dataset 
type (heading, pitch, fov for street dataset or zoom, tile_x, tile_y for map dataset). If the split 
is not recognized, it is given a default order value to ensure it is sorted after known splits. The 
function returns a new list of rows sorted according to this defined order.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, including dataset type.
rows: A list of dictionaries, where each dictionary represents a row of metadata. The keys of the dictionary
	  are column names, and the values are the corresponding cell values for that row.

Returns:
A list of dictionaries representing the input rows sorted according to the defined order based on 
split and metadata fields. The sorting is stable and consistent with the specified criteria for 
each dataset type.
'''
def sort_rows(config, rows):
	def _sort_key(row):
		split_key = SPLIT_ORDER.get(normalize_split(row.get("split", "")), 99)
		base = [split_key, numeric_sort_key(row.get("id", ""))]

		if config.name == "street":
			base.extend(
				[
					numeric_sort_key(row.get("heading", "")),
					numeric_sort_key(row.get("pitch", "")),
					numeric_sort_key(row.get("fov", "")),
				]
			)
		else:
			base.extend(
				[
					numeric_sort_key(row.get("zoom", "")),
					numeric_sort_key(row.get("tile_x", "")),
					numeric_sort_key(row.get("tile_y", "")),
				]
			)

		base.append(string_value(row.get("path", "")))
		return tuple(base)

	return sorted(rows, key=_sort_key)

'''
# ----------------------------------------------------------------------------------------------------
# reconcile_dataset()
# ----------------------------------------------------------------------------------------------------
Description:
This function performs the reconciliation of the dataset metadata by comparing the existing metadata rows
with the actual files found in the split directories. It reads the existing combined metadata and split metadata,
builds a context for existing rows, scans the split directories to find actual files and their parsed keys,
and then constructs new metadata rows for the files based on the existing context. It also handles non-file rows
based on the keep_nonfile_rows flag, deduplicates them, and combines them with the file rows. The resulting
combined metadata rows are then sorted and written back to the combined metadata CSV, and the split metadata CSVs
are also updated with the corresponding rows for each split. The function returns a summary of the reconciliation
process, including statistics about files scanned, indexed, parse errors, duplicate keys, and the number of rows
in the output.

Arguments:
config: A DatasetConfig object containing the configuration for the dataset, including paths and column names.
base_dir: A Path object representing the base directory containing the split directories and metadata files.
dry_run: A boolean flag indicating whether to perform a dry run (i.e., do not write any changes to disk).
keep_nonfile_rows: A boolean flag indicating whether to keep non-file rows from the existing combined metadata.
backup_enabled: A boolean flag indicating whether to create backup files when writing changes to disk.

Returns:
A dictionary containing a summary of the reconciliation process, including:
- "dataset": The name of the dataset.
- "dry_run": Whether the reconciliation was a dry run.
- "files_scanned": The total number of files scanned in the split directories.
- "files_indexed": The number of files that were successfully indexed with valid keys.
- "parse_errors": The number of files that could not be parsed successfully.
- "duplicate_file_keys": The number of duplicate file keys found during scanning.
- "existing_file_keys": The number of existing file keys found in the existing metadata.
- "actual_file_keys": The number of actual file keys found in the split directories.
- "kept_file_keys": The number of file keys that were present in both existing metadata and actual files.
- "added_file_keys": The number of file keys that were found in actual files but not in existing metadata.
- "removed_file_keys": The number of file keys that were present in existing metadata but not
					 in actual files.
- "invalid_existing_file_rows": The number of existing metadata rows that had invalid file keys.
- "preserved_nonfile_rows": The number of non-file rows that were preserved in the output.
- "combined_output_rows": The total number of rows in the combined output metadata.
- "split_output_rows": A dictionary mapping each split name to the number of rows in the 
					   corresponding split output metadata.
- "combined_metadata_path": The path to the combined metadata CSV file.
- "split_metadata_paths": A dictionary mapping each split name to the path of the corresponding split
					 metadata CSV file.
- "backups": A list of backup file paths that were created during the write process (if backup_enabled is True).
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# log_dataset_summary()
# ----------------------------------------------------------------------------------------------------
Description:
This function logs a summary of the dataset reconciliation process using the logging module. It takes a 
summary dictionary as input, which contains various statistics and information about the reconciliation 
process, such as the number of files scanned, indexed, parse errors, duplicate keys, existing and actual 
file keys, preserved non-file rows, combined output rows, split output rows, paths to the combined and 
split metadata files, and any backup files created. The function logs this information in a structured 
format to provide insights into the results of the reconciliation process for each dataset.

Arguments:
summary: A dictionary containing a summary of the dataset reconciliation process, 
		 including statistics and information about files scanned, indexed, 
		 parse errors, duplicate keys, existing and actual file keys, preserved non-file rows, 
		 combined output rows, split output rows, paths to metadata files, and backup files created.

Returns:
None. This function logs the summary information but does not return any value.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------------
Description:
This function serves as the main execution for the metadata reconciliation process. It parses 
command-line arguments, sets up logging, and iterates through the selected datasets 
(street, map, or both) to perform the reconciliation using the reconcile_dataset function. 
After processing each dataset, it logs a summary of the results. The function is designed to be 
run as a script, and it ensures that the reconciliation process is executed with the appropriate 
configuration and options based on the provided arguments.

Arguments:
None. This function relies on command-line arguments for configuration.

Returns:
None. This function executes the reconciliation process and logs the results but does not return any value.
'''
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

# Main entry point
if __name__ == "__main__":
	run()

	