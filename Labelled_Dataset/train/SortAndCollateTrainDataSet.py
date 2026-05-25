"""
Sorting and collating script for the training dataset.

Author: AxeWhy
Date: 25/05/2026
Last Updated: 25/05/2026

Description:
This script is responsible for sorting and collating the training dataset for the project. Due to the size limitation of
label-studio, the training dataset is split into multiple parts. This means that the output COCO JSON file from label-studio
will be split into multiple JSON files, each containing a portion of the dataset and there will be a conflict between the image IDs
and annotation IDs across the different JSON files. This script will sort and collate the multiple JSON files into a single JSON file,
ensuring that the image IDs and annotation IDs are unique across the entire dataset. The script will
also ensure that the images are correctly associated with their corresponding annotations in the final collated JSON file.

Command Line Usage:
python SortAndCollateTrainDataSet.py -i <input_directory> -o <output_json_file>
"""

import argparse
import json
import os
import sys

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_IMAGE_INDEX_CACHE = {}

"""
# ----------------------------------------------------------------------------------------------------
# _to_posix_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function converts a given file path to a POSIX-style path by replacing backslashes with forward 
slashes. This is useful for ensuring consistent path formatting across different operating systems.

Arguments:
path_text: The original file path as a string.

Returns:
The converted file path in POSIX format, where backslashes are replaced with forward slashes.
"""
def _to_posix_path(path_text):
	if isinstance(path_text, str):
		return path_text.replace("\\", "/")
	return path_text

"""
# ----------------------------------------------------------------------------------------------------
# _get_image_index()
# ----------------------------------------------------------------------------------------------------
Description:
This function retrieves an index of image file names to their absolute paths for a given root directory. It
walks through the directory tree starting from the root directory and collects all files that have extensions
matching the defined image extensions. The function uses a cache to avoid redundant indexing for the same directory.

Arguments:
root_dir: The root directory from which to start indexing image files.

Returns:
A dictionary mapping image file names (in lowercase) to their absolute paths. The keys are normalized to ensure
case-insensitive matching of file names.
"""
def _get_image_index(root_dir):
	root_abs = os.path.abspath(root_dir)
	cache_key = os.path.normcase(root_abs)
	if cache_key in _IMAGE_INDEX_CACHE:
		return _IMAGE_INDEX_CACHE[cache_key]

	index = {}
	if os.path.isdir(root_abs):
		for walk_root, dirs, files in os.walk(root_abs):
			dirs.sort()
			files.sort()
			for name in files:
				extension = os.path.splitext(name)[1].lower()
				if extension not in _IMAGE_EXTENSIONS:
					continue
				if name in index:
					continue
				index[name] = os.path.join(walk_root, name)

	_IMAGE_INDEX_CACHE[cache_key] = index
	return index

"""
# ----------------------------------------------------------------------------------------------------
# _find_existing_image_path()
# ----------------------------------------------------------------------------------------------------
Description:
This function attempts to find an existing image file path based on a given file name and a reference 
JSON directory. It generates candidate paths by considering the original file name, potential relative
paths based on the JSON directory, and common "images" subdirectory structures. It also checks against 
an indexed collection of image files under the JSON directory. The function returns the absolute path 
of the first existing image file that matches any of the candidate paths.

Arguments:
file_name: The original file name to search for.
json_dir: The directory of the JSON file, used as a reference for generating candidate paths.

Returns:
The absolute path of the existing image file if found, or None if no matching file is found.
"""
def _find_existing_image_path(file_name, json_dir):
	if not isinstance(file_name, str):
		return None

	file_name_os = file_name.replace("/", os.sep).replace("\\", os.sep)
	base_name = os.path.basename(file_name_os)
	if not base_name:
		return None

	candidate_paths = []
	if os.path.isabs(file_name_os):
		candidate_paths.append(file_name_os)
	else:
		candidate_paths.append(os.path.join(json_dir, file_name_os))

	parts = [part for part in file_name.replace("\\", "/").split("/") if part and part != "."]
	lower_parts = [part.lower() for part in parts]
	if "images" in lower_parts:
		images_idx = lower_parts.index("images")
		tail = parts[images_idx + 1 :]
		if tail:
			candidate_paths.append(os.path.join(json_dir, "images", *tail))
	candidate_paths.append(os.path.join(json_dir, "images", base_name))
	candidate_paths.append(os.path.join(os.path.dirname(json_dir), "images", base_name))

	for candidate in candidate_paths:
		if os.path.exists(candidate):
			return os.path.abspath(candidate)

	for lookup_root in (json_dir, os.path.dirname(json_dir)):
		resolved = _get_image_index(lookup_root).get(base_name)
		if resolved:
			return os.path.abspath(resolved)

	return None

'''
# ----------------------------------------------------------------------------------------------------
# _normalize_file_name()
# ----------------------------------------------------------------------------------------------------
Description:
This function normalizes the file name based on the specified mode. It can either keep the original file
name, use only the base name, or create a relative path based on the JSON directory and a specified root.

Arguments:
file_name: The original file name to be normalized.
mode: The mode of normalization. It can be "keep", "basename", or "relative
json_dir: The directory of the JSON file, used for relative path mode.
relative_root: The root directory for calculating relative paths, used for relative path mode.

Returns:
The normalized file name based on the specified mode.

'''
def _normalize_file_name(file_name, mode, json_dir, relative_root):
	if not isinstance(file_name, str):
		return file_name

	if mode == "keep":
		return _to_posix_path(file_name)

	base_name = os.path.basename(file_name)
	if not base_name:
		return _to_posix_path(file_name)

	if mode == "basename":
		return _to_posix_path(base_name)

	if mode == "relative":
		image_path = _find_existing_image_path(file_name, json_dir)
		if image_path is None:
			image_path = os.path.join(json_dir, "images", base_name)
		return _to_posix_path(os.path.relpath(image_path, relative_root))

	return _to_posix_path(file_name)

'''
# ----------------------------------------------------------------------------------------------------
# _next_category_id()
# ----------------------------------------------------------------------------------------------------
Description:
This function calculates the next available category ID based on the existing categories. It checks the
maximum ID among the existing categories and returns the next ID that can be used for a new category.

Arguments:
categories: A list of existing category dictionaries, each containing an "id" field.

Returns:
The next available category ID, which is one greater than the maximum existing ID
If there are no existing categories, it returns 0.

'''
def _next_category_id(categories):
	if not categories:
		return 0
	return max(cat.get("id", -1) for cat in categories) + 1

'''
# ----------------------------------------------------------------------------------------------------
# _update_categories()
# ----------------------------------------------------------------------------------------------------
Description:
This function updates the canonical category list with categories from a new JSON file. It checks for
existing categories by name and either maps to the existing category ID or creates a new category with
a unique ID. It returns a mapping from the original category IDs in the new JSON file to the canonical
category IDs.

Arguments:
categories: A list of category dictionaries from the new JSON file, each containing at least a "name" field.
canonical_by_name: A dictionary mapping category names to their canonical category dictionaries.
canonical_categories: A list of canonical category dictionaries that have been collected so far.

Returns:
A dictionary mapping the original category IDs to the canonical category IDs.

'''
def _update_categories(categories, canonical_by_name, canonical_categories):
	file_category_map = {}
	for category in categories or []:
		name = category.get("name")
		if not name:
			continue

		if name in canonical_by_name:
			canonical = canonical_by_name[name]
			file_category_map[category.get("id")] = canonical.get("id")
			continue

		new_category = dict(category)
		new_id = new_category.get("id")
		existing_ids = {cat.get("id") for cat in canonical_categories}
		if new_id in existing_ids or new_id is None:
			new_id = _next_category_id(canonical_categories)

		new_category["id"] = new_id
		canonical_categories.append(new_category)
		canonical_by_name[name] = new_category
		file_category_map[category.get("id")] = new_id

	return file_category_map

'''
# ----------------------------------------------------------------------------------------------------
# _discover_result_files()
# ----------------------------------------------------------------------------------------------------
Description:
This function discovers all JSON files in the specified input directory that match the given filename
pattern. It walks through the directory tree and allocates the paths of the matching files, while also
ensuring that the output JSON file (if specified) is not included in the list of result files. The function
returns a sorted list of the discovered JSON file paths.

Arguments:
input_dir: The directory to search for JSON files.
pattern: The filename pattern to match.
output_json: The path to the output JSON file (if specified).

Returns:
A sorted list of the discovered JSON file paths.

'''
def _discover_result_files(input_dir, pattern, output_json):
	output_abs = None
	if output_json:
		output_abs = os.path.normcase(os.path.abspath(output_json))

	result_files = []
	for root, _, files in os.walk(input_dir):
		if pattern not in files:
			continue
		path = os.path.normcase(os.path.abspath(os.path.join(root, pattern)))
		if output_abs and path == output_abs:
			continue
		result_files.append(path)

	result_files.sort()
	return result_files

'''
# ----------------------------------------------------------------------------------------------------
# _load_json()
# ----------------------------------------------------------------------------------------------------
Description:
This function loads a JSON file from the specified path and returns the parsed data. It opens the file
with UTF-8 encoding and uses the json module to parse the contents of the file. The function returns the
resulting data structure, which is typically a dictionary representing the JSON content.

Arguments:
path: The file path to the JSON file that needs to be loaded.

Returns:
The parsed JSON data as a Python dictionary.

'''
def _load_json(path):
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)

'''
# ----------------------------------------------------------------------------------------------------
# collate_coco_json()
# ----------------------------------------------------------------------------------------------------
Description:
This function collates multiple COCO JSON files into a single dataset. It processes each JSON file to 
extract images, annotations, and categories, while ensuring that category IDs are consistent across files.
It also normalizes image file names based on the specified mode and sorts images and annotations if required.
The function returns the collated dataset as a dictionary and statistics about any skipped images or annotations.

Arguments:
result_files: A list of file paths to the COCO JSON files that need to be collated.
file_name_mode: The mode for normalizing image file names ("keep", "basename", or "relative").
sort_images: A boolean indicating whether to sort images by file name.
relative_root: The root directory for calculating relative paths, used for relative file name mode.

Returns:
The collated dataset as a dictionary containing "images", "categories", and "annotations" keys,
along with a statistics dictionary that includes counts of skipped images, skipped annotations, and
missing image references in annotations.

'''
def collate_coco_json(result_files, file_name_mode, sort_images, relative_root):
	canonical_categories = []
	canonical_by_name = {}
	image_records = []
	annotation_records = []

	info_block = None
	licenses_block = None

	image_index = 0
	annotation_index = 0
	skipped_images = 0
	skipped_annotations = 0

	for path in result_files:
		data = _load_json(path)
		if info_block is None and "info" in data:
			info_block = data["info"]
		if licenses_block is None and "licenses" in data:
			licenses_block = data["licenses"]

		file_category_map = _update_categories(
			data.get("categories", []), canonical_by_name, canonical_categories
		)

		json_dir = os.path.dirname(path)
		for image in data.get("images", []):
			original_id = image.get("id")
			if original_id is None:
				skipped_images += 1
				continue
			image_copy = dict(image)
			image_copy["file_name"] = _normalize_file_name(
				image_copy.get("file_name"), file_name_mode, json_dir, relative_root
			)
			image_records.append(
				{
					"source_key": (path, original_id),
					"source_index": image_index,
					"image": image_copy,
				}
			)
			image_index += 1

		for annotation in data.get("annotations", []):
			original_id = annotation.get("id")
			image_id = annotation.get("image_id")
			if image_id is None:
				skipped_annotations += 1
				continue
			annotation_copy = dict(annotation)
			category_id = annotation_copy.get("category_id")
			if category_id in file_category_map:
				annotation_copy["category_id"] = file_category_map[category_id]
			annotation_records.append(
				{
					"source_key": (path, original_id),
					"source_index": annotation_index,
					"image_key": (path, image_id),
					"annotation": annotation_copy,
				}
			)
			annotation_index += 1

	if sort_images:
		image_records.sort(
			key=lambda record: (
				str(record["image"].get("file_name", "")).lower(),
				record["source_index"],
			)
		)

	image_id_map = {}
	for new_id, record in enumerate(image_records):
		record["image"]["id"] = new_id
		image_id_map[record["source_key"]] = new_id

	valid_annotations = []
	missing_image_refs = 0
	for record in annotation_records:
		image_id = image_id_map.get(record["image_key"])
		if image_id is None:
			missing_image_refs += 1
			continue
		record["annotation"]["image_id"] = image_id
		valid_annotations.append(record)

	if sort_images:
		valid_annotations.sort(
			key=lambda record: (
				record["annotation"].get("image_id", -1),
				record["source_index"],
			)
		)
	else:
		valid_annotations.sort(key=lambda record: record["source_index"])

	for new_id, record in enumerate(valid_annotations):
		record["annotation"]["id"] = new_id

	output = {
		"images": [record["image"] for record in image_records],
		"categories": sorted(
			canonical_categories, key=lambda cat: cat.get("id", -1)
		),
		"annotations": [record["annotation"] for record in valid_annotations],
	}

	if info_block is not None:
		output["info"] = info_block
	if licenses_block is not None:
		output["licenses"] = licenses_block

	stats = {
		"skipped_images": skipped_images,
		"skipped_annotations": skipped_annotations,
		"missing_image_refs": missing_image_refs,
	}

	return output, stats

'''
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
Description:
This function parses command-line arguments for the script. It sets up an argument parser with options
for specifying the input directory, output JSON file, filename pattern, file name normalization mode, 
and whether to sort images. The function returns the parsed arguments as a namespace object.

Arguments:
None. This function does not take any arguments directly, but it processes command-line arguments
when the script is executed.

Returns:
None. The function returns a namespace object containing the parsed command-line arguments.

'''
def parse_args():
	parser = argparse.ArgumentParser(
		description="Collate multiple COCO JSON exports into a single dataset."
	)
	default_input = os.path.dirname(os.path.abspath(__file__))
	parser.add_argument(
		"-i",
		"--input-dir",
		default=default_input,
		help="Root folder containing label-studio export subfolders.",
	)
	parser.add_argument(
		"-o",
		"--output-json",
		default=None,
		help="Output JSON file path. Defaults to <input-dir>/result.json.",
	)
	parser.add_argument(
		"--pattern",
		default="result.json",
		help="Filename pattern to search for (default: result.json).",
	)
	parser.add_argument(
		"--file-name-mode",
		choices=["keep", "basename", "relative"],
		default="relative",
		help=(
			"How to rewrite image file_name fields in the output JSON "
			"(default: relative)."
		),
	)
	parser.add_argument(
		"--no-sort",
		dest="sort_images",
		action="store_false",
		help="Preserve input ordering instead of sorting images by file name.",
	)
	parser.set_defaults(sort_images=True)
	return parser.parse_args()

'''
# ----------------------------------------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------------------------------------
Description:
This is the main execution function for the script. It parses command-line arguments, discovers relevant
JSON files in the input directory, collates the data from those files into a single dataset, and writes
the output to a specified JSON file. It also prints statistics about the collation process, including
the number of files collated, the total number of images, annotations, and categories in the output, and
any skipped images or annotations due to missing IDs.

Arguments:
None. This function does not take any arguments directly, but it processes command-line arguments

Returns:
None. The function returns an exit code (0 for success, 1 for failure) based on the outcome of the execution.

'''
def main():
	args = parse_args()
	input_dir = os.path.abspath(args.input_dir)
	output_json = args.output_json
	if output_json is None:
		output_json = os.path.join(input_dir, "result.json")
	output_json = os.path.abspath(output_json)

	result_files = _discover_result_files(input_dir, args.pattern, output_json)
	if not result_files:
		pattern_path = os.path.join(input_dir, args.pattern)
		if os.path.normcase(os.path.abspath(pattern_path)) == os.path.normcase(output_json) and os.path.exists(output_json):
			print("No source JSON files found. Reusing the existing output JSON for normalization.")
			result_files = [output_json]
		else:
			print("No matching JSON files found.")
			return 1

	output, stats = collate_coco_json(
		result_files, args.file_name_mode, args.sort_images, input_dir
	)

	os.makedirs(os.path.dirname(output_json), exist_ok=True)
	with open(output_json, "w", encoding="utf-8") as handle:
		json.dump(output, handle, indent=2)

	resolved_images = 0
	total_images = 0
	for image in output.get("images", []):
		file_name = image.get("file_name")
		if not isinstance(file_name, str):
			continue
		total_images += 1
		candidate = file_name if os.path.isabs(file_name) else os.path.join(input_dir, file_name)
		if os.path.exists(candidate):
			resolved_images += 1

	print(f"Collated {len(result_files)} files into: {output_json}")
	print(
		f"Images: {len(output['images'])} | "
		f"Annotations: {len(output['annotations'])} | "
		f"Categories: {len(output['categories'])}"
	)
	if total_images and resolved_images != total_images:
		print(
			"Warning: "
			f"{total_images - resolved_images}/{total_images} image file_name paths "
			"do not resolve under the input directory. "
			"Consider using --file-name-mode relative."
		)
	if stats["skipped_images"]:
		print(f"Skipped images without id: {stats['skipped_images']}")
	if stats["skipped_annotations"]:
		print(f"Skipped annotations without image_id: {stats['skipped_annotations']}")
	if stats["missing_image_refs"]:
		print(
			"Skipped annotations with missing image refs: "
			f"{stats['missing_image_refs']}"
		)

	return 0

# Main entry point
if __name__ == "__main__":
	sys.exit(main())
