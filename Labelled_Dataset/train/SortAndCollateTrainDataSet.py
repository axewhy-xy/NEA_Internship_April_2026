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