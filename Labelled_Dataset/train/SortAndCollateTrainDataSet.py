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
		default="keep",
		help="How to rewrite image file_name fields in the output JSON.",
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
		print("No matching JSON files found.")
		return 1

	output, stats = collate_coco_json(
		result_files, args.file_name_mode, args.sort_images, input_dir
	)

	os.makedirs(os.path.dirname(output_json), exist_ok=True)
	with open(output_json, "w", encoding="utf-8") as handle:
		json.dump(output, handle, indent=2)

	print(f"Collated {len(result_files)} files into: {output_json}")
	print(
		f"Images: {len(output['images'])} | "
		f"Annotations: {len(output['annotations'])} | "
		f"Categories: {len(output['categories'])}"
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