"""
Dataset extraction script for Google Maps Street View images.

Author: AxeWhy
Date: 2026-04-14
Last updated: 2026-04-15

Description:
Builds a Street View image dataset from a CSV of locations and labels.
Supports both coordinate and postal-code input formats, performs
train/validation/test splitting, downloads images for multiple headings,
and writes detailed metadata logs.

Input CSV formats supported:
1) id,lat,lon,label
2) id,postal_code,label

Usage example:
python Dataset/Extraction_Street.py --input-csv Dataset/hdb_points_postal.csv

API key:
Set GOOGLE_MAPS_API_KEY in your environment, or pass --api-key.
"""

import argparse
import csv
import logging
import os
import random
import re
import time
from pathlib import Path

import requests

# -----------------------------
# Defaults / constants
# -----------------------------
DEFAULT_INPUT_CSV = "Dataset/Input/hdb_points_postal.csv"
DEFAULT_OUTPUT_ROOT = Path("Dataset/Street")
DEFAULT_COMBINED_METADATA = "street_metadata.csv"

DEFAULT_RANDOM_SEED = 42
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALID_RATIO = 0.15

DEFAULT_HEADINGS = (0, 90, 180, 270)
DEFAULT_PITCH = 0
DEFAULT_FOV = 90
DEFAULT_SIZE = "640x640"
DEFAULT_RADIUS = 50
DEFAULT_SOURCE = "outdoor"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.2
DEFAULT_MIN_DELAY_SECONDS = 0.2
DEFAULT_GEOCODE_DELAY_SECONDS = 0.2

RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
ONEMAP_GEOCODE_URL = "https://www.onemap.gov.sg/api/common/elastic/search"

METADATA_COLUMNS = [
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
]

'''
------------------------------
parse_args()
------------------------------
Description:
Defines and parse command-line arguments for the dataset extraction script.
Supports configuration of input/output paths, API key, dataset split ratios,
image parameters, and request retry behavior.

Arguments:
--input-csv: Path to input CSV file with location samples.
--output-root: Root directory for output dataset and metadata.
--metadata-csv: Filename for combined metadata CSV in output root.
--api-key: Google Maps API key (can also be set via GOOGLE_MAPS_API_KEY env var).
--seed: Random seed for reproducible train/valid/test splitting.
--train-ratio: Proportion of samples to use for training set.
--valid-ratio: Proportion of samples to use for validation set.
--headings: Comma-separated list of headings (in degrees) for image capture.
--pitch: Pitch angle for Street View images.
--fov: Field of view for Street View images.
--size: Image size in WIDTHxHEIGHT format, e.g. 640x640.
--radius: Search radius (in meters) for Street View metadata lookup.
--source: Image source for Street View (e.g. "outdoor").
--timeout-seconds: Request timeout in seconds.
--max-retries: Maximum number of retries for failed requests.
--retry-backoff-seconds: Base seconds for exponential backoff between retries.
--min-delay-seconds: Minimum delay between requests to avoid rate limits.
--geocode-delay-seconds: Delay between geocoding requests when postal codes are used.
--metadata-only: If set, only query metadata without downloading images.

Usage:
python Dataset/Extraction_Street.py --input-csv Dataset/hdb_points_postal.csv
'''
def parse_args():
	parser = argparse.ArgumentParser(
		description="Extract Google Street View dataset with metadata and split folders."
	)
	parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
	parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
	parser.add_argument("--metadata-csv", default=DEFAULT_COMBINED_METADATA)
	parser.add_argument("--api-key", default=os.getenv("GOOGLE_MAPS_API_KEY"))

	parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
	parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
	parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)

	parser.add_argument("--headings", default=",".join(map(str, DEFAULT_HEADINGS)))
	parser.add_argument("--pitch", type=int, default=DEFAULT_PITCH)
	parser.add_argument("--fov", type=int, default=DEFAULT_FOV)
	parser.add_argument("--size", default=DEFAULT_SIZE)
	parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
	parser.add_argument("--source", default=DEFAULT_SOURCE, choices=["outdoor", "default"])

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
		"--metadata-only",
		action="store_true",
		help="Only query metadata (availability), skip image downloads.",
	)
	return parser.parse_args()

'''
------------------------------
setup_logging()
------------------------------
Description:
Configures logging to output timestamped messages to the console.
Sets log level to INFO and defines a consistent log message format.
'''
def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

'''
------------------------------
validate_args(args)
------------------------------
Description:
Validates the command-line arguments for correctness and consistency.
Checks for presence of API key, valid split ratios, correct size format, and valid FOV range.
Raises ValueError with descriptive messages if any validation checks fail.

Arguments:
args: Parsed command-line arguments to validate.
'''
def validate_args(args):
	if not args.api_key:
		raise ValueError(
			"Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key."
		)
	if args.train_ratio <= 0 or args.valid_ratio < 0:
		raise ValueError("Split ratios must be non-negative and train > 0.")
	if args.train_ratio + args.valid_ratio >= 1.0:
		raise ValueError("train_ratio + valid_ratio must be < 1.0.")
	if not re.fullmatch(r"\d+x\d+", args.size):
		raise ValueError("--size must be in WIDTHxHEIGHT format, e.g. 640x640.")
	if not (0 <= args.fov <= 120):
		raise ValueError("--fov must be between 0 and 120.")

'''
------------------------------
parse_headings(headings_text)
------------------------------
Description:
Parses a comma-separated string of headings into a list of unique integers.

Arguments:
headings_text: A string containing comma-separated heading values.

Returns:
A list of unique heading values as integers.

Raises: 
ValueError if no valid headings are found.
'''
def parse_headings(headings_text):
	values = []
	for token in headings_text.split(","):
		token = token.strip()
		if not token:
			continue
		heading = int(token)
		heading = heading % 360
		values.append(heading)
	if not values:
		raise ValueError("At least one heading is required.")
	# Preserve insertion order while deduplicating.
	return list(dict.fromkeys(values))

'''
------------------------------
_row_value(row, *keys)
------------------------------
Description:
Helper function to extract the first non-empty value from a dictionary row based on multiple possible keys.
Iterates through the provided keys in order, checks if the key exists in the row, and
returns the first non-empty value found. If no valid value is found, returns None.

Arguments:
row: A dictionary representing a row of data.
*keys: A variable number of string keys to check in the row.

Returns:
The first non-empty value found for the provided keys, or None if no valid value is found.
'''
def _row_value(row, *keys):
	for key in keys:
		value = row.get(key)
		if value is None:
			continue
		text = str(value).strip()
		if text:
			return text
	return None

'''
------------------------------
request_with_retries(session, method, url, *, params=None, timeout=30, max_retries=4, backoff_seconds=1.2)
------------------------------
Description:
Performs an HTTP request with retries and exponential backoff for transient errors.
Uses the provided requests.Session to make the request, and retries on specified HTTP status codes or request
exceptions. Implements exponential backoff between retries to mitigate rate limits and transient issues.

Arguments:
session: A requests.Session object to use for making the HTTP request.
method: HTTP method to use (e.g. "GET", "POST").
url: The URL to which the request is sent.
params: Optional dictionary of query parameters to include in the request.
timeout: The timeout for the HTTP request.
max_retries: The maximum number of retry attempts.
backoff_seconds: The base number of seconds to wait between retries.

Returns:
The HTTP response object if the request is successful.

Raises:
RuntimeError if the request fails after all retry attempts, with the last encountered error as context.
'''
def request_with_retries(
	session,
	method,
	url,
	*,
	params=None,
	timeout=30,
	max_retries=4,
	backoff_seconds=1.2,
):
	last_error = None
	for attempt in range(max_retries + 1):
		try:
			response = session.request(method, url, params=params, timeout=timeout)

			if response.status_code in RETRY_HTTP_STATUSES and attempt < max_retries:
				wait_seconds = backoff_seconds ** (attempt + 1)
				logging.warning(
					"HTTP %s from %s. Retrying in %.2fs (%s/%s).",
					response.status_code,
					url,
					wait_seconds,
					attempt + 1,
					max_retries,
				)
				time.sleep(wait_seconds)
				continue

			response.raise_for_status()
			return response
		except requests.RequestException as exc:
			last_error = exc
			if attempt >= max_retries:
				break
			wait_seconds = backoff_seconds ** (attempt + 1)
			logging.warning(
				"Request error on %s. Retrying in %.2fs (%s/%s): %s",
				url,
				wait_seconds,
				attempt + 1,
				max_retries,
				exc,
			)
			time.sleep(wait_seconds)

	raise RuntimeError(f"Request failed after retries: {url}") from last_error

'''
-------------------------------
geocode_postal_sg(session, postal_code, timeout, max_retries, backoff_seconds)
-------------------------------
Desciription:
Geocodes a Singapore postal code using the OneMap API to obtain latitude and longitude coordinates.
Sends a GET request to the OneMap geocoding endpoint with the postal code as a
query parameter, and parses the response to extract the first result's latitude and longitude.
Implements retries with exponential backoff for transient errors.

Arguments:
session: A requests.Session object to use for making the HTTP request.
postal_code: The Singapore postal code to geocode.
timeout: The timeout for the HTTP request.
max_retries: The maximum number of retry attempts for the geocoding request.
backoff_seconds: The base number of seconds to wait between retries.

Returns:
A tuple of (latitude, longitude) if geocoding is successful, or (None, None) if geocoding fails or no results are found.
'''
def geocode_postal_sg(
	session,
	postal_code,
	timeout,
	max_retries,
	backoff_seconds,
):
	postal = str(postal_code).strip()
	if not postal:
		return None, None

	params = {
		"searchVal": postal,
		"returnGeom": "Y",
		"getAddrDetails": "Y",
		"pageNum": 1,
	}
	response = request_with_retries(
		session,
		"GET",
		ONEMAP_GEOCODE_URL,
		params=params,
		timeout=timeout,
		max_retries=max_retries,
		backoff_seconds=backoff_seconds,
	)
	payload = response.json()

	if int(payload.get("found", 0)) == 0:
		return None, None

	results = payload.get("results") or []
	if not results:
		return None, None

	first = results[0]
	try:
		lat = float(first["LATITUDE"])
		lon = float(first["LONGITUDE"])
	except (KeyError, TypeError, ValueError):
		return None, None
	return lat, lon

'''
-------------------------------
load_samples(csv_path, session, timeout, max_retries, backoff_seconds, geocode_delay_seconds)
-------------------------------
Description:
loads location samples from a CSV file, supporting both coordinate and postal code formats.
for rows with postal codes, performs geocoding to obtain lat/lon coordinates using the OneMap API.
Iterates through each row in the CSV, extracts relevant fields, and handles missing or invalid data.
Returns a list of valid samples with their coordinates, labels, and metadata for further processing.

Arguments:
csv_path: Path to the input CSV file containing location samples.
session: A requests.Session object to use for making HTTP requests (e.g. for geocoding).
timeout: The timeout for HTTP requests.
max_retries: The maximum number of retry attempts for HTTP requests.
backoff_seconds: The base number of seconds to wait between retries for HTTP requests.
geocode_delay_seconds: The delay in seconds between geocoding requests to avoid rate limits.

Returns:
A list of dictionaries, each representing a valid sample with keys: "id", "lat",
"lon", "label", and "postal_code". Raises ValueError if no valid samples are loaded.
'''
def load_samples(
	csv_path,
	session,
	timeout,
	max_retries,
	backoff_seconds,
	geocode_delay_seconds,
):
	rows = []
	postal_cache = {}

	with open(csv_path, newline="", encoding="utf-8") as file_obj:
		reader = csv.DictReader(file_obj)

		for line_no, row in enumerate(reader, start=2):
			sample_id = _row_value(row, "id", "ID") or f"row_{line_no}"
			label = _row_value(row, "label", "Label") or ""

			lat_raw = _row_value(row, "lat", "latitude", "LAT", "Latitude")
			lon_raw = _row_value(row, "lon", "lng", "longitude", "LON", "Longitude")
			postal_code = _row_value(
				row,
				"postal_code",
				"postal",
				"postcode",
				"postal code",
				"POSTAL",
			)

			lat = None
			lon = None

			if lat_raw is not None and lon_raw is not None:
				try:
					lat = float(lat_raw)
					lon = float(lon_raw)
				except ValueError:
					logging.warning(
						"Skipping line %s (%s): invalid lat/lon values.",
						line_no,
						sample_id,
					)
					continue
			elif postal_code:
				if postal_code not in postal_cache:
					lat_lon = geocode_postal_sg(
						session,
						postal_code,
						timeout=timeout,
						max_retries=max_retries,
						backoff_seconds=backoff_seconds,
					)
					postal_cache[postal_code] = lat_lon
					time.sleep(geocode_delay_seconds)

				lat, lon = postal_cache[postal_code]
				if lat is None or lon is None:
					logging.warning(
						"Skipping line %s (%s): could not geocode postal code %s.",
						line_no,
						sample_id,
						postal_code,
					)
					continue
			else:
				logging.warning(
					"Skipping line %s (%s): missing both lat/lon and postal code.",
					line_no,
					sample_id,
				)
				continue

			rows.append(
				{
					"id": sample_id,
					"lat": lat,
					"lon": lon,
					"label": label,
					"postal_code": postal_code or "",
				}
			)

	if not rows:
		raise ValueError(f"No valid samples loaded from {csv_path}.")
	return rows

'''
-------------------------------
split_rows(rows, seed, train_ratio, valid_ratio)
-------------------------------
Description:
Splits a list of rows into train, validation, and test sets based on specified ratios.
Shuffles the rows using a random seed for reproducibility, then calculates the number of
samples for each split and returns the corresponding subset of rows for train, validation,
and test. Ensures that the splits are mutually exclusive and collectively exhaustive.

Arguments:
rows: A list of data rows to be split.
seed: A random seed for shuffling the rows to ensure reproducibility.
train_ratio: The proportion of rows to include in the training set (e.g. 0.7).
valid_ratio: The proportion of rows to include in the validation set (e.g. 0.15).

Returns:
A tuple of three lists: (train_rows, valid_rows, test_rows) corresponding to the
training, validation, and test splits of the input rows.
'''
def split_rows(rows, seed, train_ratio, valid_ratio):
	rng = random.Random(seed)
	rows_copy = rows[:]
	rng.shuffle(rows_copy)

	count = len(rows_copy)
	n_train = int(count * train_ratio)
	n_valid = int(count * valid_ratio)

	train = rows_copy[:n_train]
	valid = rows_copy[n_train : n_train + n_valid]
	test = rows_copy[n_train + n_valid :]
	return train, valid, test

'''
-------------------------------
get_streetview_metadata(session, api_key, lat, lon, radius, 
                        source, timeout, max_retries, backoff_seconds)
-------------------------------
Description:
Queries the Google Street View Metadata API for a specific location defined by latitude and longitude.
Send a GET request to the metadat endpoint with the provided parameters, including search radius and 
image source. Implements retries with exponential backoff for transient errors. Parses and returns the
JSON response containing metadata about the Street View imagery available at the specified location.

Arguments:
session: A requests.Session object to use for making the HTTP request.
api_key: The Google Maps API key for authentication.
lat: The latitude of the location to query.
lon: The longitude of the location to query.
radius: The search radius (in meters) for finding Street View imagery around the specified location.
source: The image source for Street View (e.g. "outdoor").
timeout: The timeout for the HTTP request.
max_retries: The maximum number of retry attempts for the HTTP request.
backoff_seconds: The base number of seconds to wait between retries for the HTTP request.

Returns:
A dictionary containing the parsed JSON response from the Street View Metadata API, which includes information about the
availability and details of Street View imagery at the specified location.
'''
def get_streetview_metadata(
	session,
	api_key,
	lat,
	lon,
	radius,
	source,
	timeout,
	max_retries,
	backoff_seconds,
):
	params = {
		"location": f"{lat},{lon}",
		"radius": radius,
		"source": source,
		"key": api_key,
	}
	response = request_with_retries(
		session,
		"GET",
		METADATA_URL,
		params=params,
		timeout=timeout,
		max_retries=max_retries,
		backoff_seconds=backoff_seconds,
	)
	return response.json()

'''
-------------------------------
download_streetview_image(session, api_key, *, out_path, heading, 
                          pitch, fov, size, source, timeout, max_retries, 
						  backoff_seconds, pano_id=None, lat=None, lon=None)
-------------------------------
Description:
Downloads a Street View image from the Google Street View API based on either a 
pano_id or a location defined by latitude and longitude. Constructs the appropriate
query parameters for the image request, including heading, pitch, field of view, and size.
Implements retries with exponential backoff for transient errors. Saves the downloaded image
to the specified output path.

Arguments:
session: A requests.Session object to use for making the HTTP request.
api_key: The Google Maps API key for authentication.
out_path: The file path where the downloaded image should be saved.
heading: The compass heading (in degrees) for the Street View image.
pitch: The pitch angle for the Street View image.
fov: The field of view for the Street View image.
size: The size of the Street View image in WIDTHxHEIGHT format (e.g. "640x640").
source: The image source for Street View (e.g. "outdoor").
timeout: The timeout for the HTTP request.
max_retries: The maximum number of retry attempts for the HTTP request.
backoff_seconds: The base number of seconds to wait between retries for the HTTP request.
pano_id: (Optional) The panorama ID to specify for the image request. If provided, lat/lon are not required.
lat: (Optional) The latitude of the location for the image request. Required if pano_id
is not provided.
lon: (Optional) The longitude of the location for the image request. Required if pano_id
is not provided.

Returns:
None. The function saves the downloaded image to the specified output path. Raises an exception if the
download fails after retries.
'''
def download_streetview_image(
	session,
	api_key,
	*,
	out_path,
	heading,
	pitch,
	fov,
	size,
	source,
	timeout,
	max_retries,
	backoff_seconds,
	pano_id=None,
	lat=None,
	lon=None,
):
	params = {
		"size": size,
		"heading": heading,
		"pitch": pitch,
		"fov": fov,
		"source": source,
		"return_error_code": "true",
		"key": api_key,
	}

	if pano_id:
		params["pano"] = pano_id
	elif lat is not None and lon is not None:
		params["location"] = f"{lat},{lon}"
	else:
		raise ValueError("Either pano_id or lat/lon must be provided for image download.")

	response = request_with_retries(
		session,
		"GET",
		IMAGE_URL,
		params=params,
		timeout=timeout,
		max_retries=max_retries,
		backoff_seconds=backoff_seconds,
	)
	out_path.write_bytes(response.content)

'''
-------------------------------
sanitize_filename(value)
-------------------------------
Description:
Sanitizes a string value to be safe for use as a filename by replacing any
characters that are not alphanumeric, dot, underscore, or hyphen with an underscore.
This helps ensure that generated filenames for downloaded images do not contain
invalid or problematic characters.

Arguments:
value: The input value to sanitize, which can be of any type but will be converted to a string.

Returns:
A sanitized string that is safe for use as a filename.
'''
def sanitize_filename(value):
	return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))

'''
-------------------------------
write_metadata_row(writer, row)
-------------------------------
Description:
Writes a row of metadata to a CSV file using the provided CSV writer object.
Extracts the values for the predefined METADATA_COLUMNS from the input row dictionary,
ensuring that the columns are written in a consistent order. If a column is missing from the row,
an empty string is written for that column. This function helps maintain a structured and consistent
format for the metadata CSV files that log the details of each sample, including any errors encountered
during metadata retrieval or image downloading.

Arguments:
writer: A csv.writer object that is used to write rows to a CSV file.
row: A dictionary containing the metadata for a sample, which may include keys corresponding to the METADATA_COLUMNS.

Returns:
None. The function writes a single row to the CSV file using the provided writer.
'''
def write_metadata_row(writer, row):
	writer.writerow([row.get(col, "") for col in METADATA_COLUMNS])

'''
-------------------------------
export_split(rows, split_dir, split_name, *, session, api_key, headings, pitch, fov,
			 size, radius, source, timeout, max_retries, backoff_seconds, 
			 min_delay_seconds, metadata_only, combined_writer, split_writer)
-------------------------------
Description:
Processes a list of rows for a specific dataset split (train/valid/test) by querying
Street View metadata and downloading images for each sample. For each row, it retrieves
the metadata for the specified location, checks the availability of Street View imagery,
and if available, downloads images for the specified headings. It logs the results in
the combined metadata CSV and the split-specific metadata CSV, including any errors
encountered during metadata retrieval or image downloading. The function also
maintains counters for the number of samples processed, metadata successes/failures,
and image download successes/failures, which are returned at the end for summary statistics.

Arguments:
rows: A list of dictionaries representing the samples to process for this split.
split_dir: The directory where downloaded images for this split should be saved.
split_name: The name of the split (e.g. "train", "valid", "test") for logging purposes.
session: A requests.Session object to use for making HTTP requests.
api_key: The Google Maps API key for authentication.
headings: A list of heading values (in degrees) for which to download Street View images.
pitch: The pitch angle for the Street View images.
fov: The field of view for the Street View images.
size: The size of the Street View images in WIDTHxHEIGHT format (e.g. "640x640").
radius: The search radius (in meters) for Street View metadata lookup.
source: The image source for Street View (e.g. "outdoor").
timeout: The timeout for HTTP requests.
max_retries: The maximum number of retry attempts for HTTP requests.
backoff_seconds: The base number of seconds to wait between retries for HTTP requests.
min_delay_seconds: The minimum delay in seconds between processing each sample to avoid rate limits.
metadata_only: If True, only query metadata without downloading images.
combined_writer: A csv.writer object for writing to the combined metadata CSV file.
split_writer: A csv.writer object for writing to the split-specific metadata CSV file.

Returns:
A dictionary containing counters for the number of samples processed, 
metadata successes/failures, and image download successes/failures.
'''
def export_split(
	rows,
	split_dir,
	split_name,
	*,
	session,
	api_key,
	headings,
	pitch,
	fov,
	size,
	radius,
	source,
	timeout,
	max_retries,
	backoff_seconds,
	min_delay_seconds,
	metadata_only,
	combined_writer,
	split_writer,
):
	counters = {
		"samples": len(rows),
		"metadata_ok": 0,
		"metadata_not_ok": 0,
		"image_ok": 0,
		"image_failed": 0,
	}

	for row in rows:
		sample_id = row["id"]
		lat = row["lat"]
		lon = row["lon"]
		label = row["label"]
		postal_code = row.get("postal_code", "")

		try:
			metadata = get_streetview_metadata(
				session,
				api_key,
				lat,
				lon,
				radius=radius,
				source=source,
				timeout=timeout,
				max_retries=max_retries,
				backoff_seconds=backoff_seconds,
			)
			status = metadata.get("status", "UNKNOWN")
			pano_id = metadata.get("pano_id", "")
			capture_date = metadata.get("date", "")
		except Exception as exc:
			status = "METADATA_FAILED"
			pano_id = ""
			capture_date = ""
			counters["metadata_not_ok"] += 1

			error_row = {
				"id": sample_id,
				"lat": lat,
				"lon": lon,
				"postal_code": postal_code,
				"label": label,
				"split": split_name,
				"heading": "",
				"pitch": pitch,
				"fov": fov,
				"size": size,
				"pano_id": pano_id,
				"capture_date": capture_date,
				"status": status,
				"path": "",
				"error": str(exc),
			}
			write_metadata_row(combined_writer, error_row)
			write_metadata_row(split_writer, error_row)
			logging.warning("%s | metadata failed: %s", sample_id, exc)
			time.sleep(min_delay_seconds)
			continue

		if status != "OK":
			counters["metadata_not_ok"] += 1
			no_image_row = {
				"id": sample_id,
				"lat": lat,
				"lon": lon,
				"postal_code": postal_code,
				"label": label,
				"split": split_name,
				"heading": "",
				"pitch": pitch,
				"fov": fov,
				"size": size,
				"pano_id": pano_id,
				"capture_date": capture_date,
				"status": status,
				"path": "",
				"error": "",
			}
			write_metadata_row(combined_writer, no_image_row)
			write_metadata_row(split_writer, no_image_row)
			logging.info("%s | no imagery status=%s", sample_id, status)
			time.sleep(min_delay_seconds)
			continue

		counters["metadata_ok"] += 1

		if metadata_only:
			meta_only_row = {
				"id": sample_id,
				"lat": lat,
				"lon": lon,
				"postal_code": postal_code,
				"label": label,
				"split": split_name,
				"heading": "",
				"pitch": pitch,
				"fov": fov,
				"size": size,
				"pano_id": pano_id,
				"capture_date": capture_date,
				"status": "OK_METADATA_ONLY",
				"path": "",
				"error": "",
			}
			write_metadata_row(combined_writer, meta_only_row)
			write_metadata_row(split_writer, meta_only_row)
			time.sleep(min_delay_seconds)
			continue

		safe_id = sanitize_filename(sample_id)
		for heading in headings:
			file_name = f"{safe_id}_{split_name}_h{heading}_p{pitch}_f{fov}.jpg"
			out_path = split_dir / file_name

			try:
				download_streetview_image(
					session,
					api_key,
					out_path=out_path,
					heading=heading,
					pitch=pitch,
					fov=fov,
					size=size,
					source=source,
					timeout=timeout,
					max_retries=max_retries,
					backoff_seconds=backoff_seconds,
					pano_id=pano_id,
					lat=lat,
					lon=lon,
				)
				counters["image_ok"] += 1
				row_data = {
					"id": sample_id,
					"lat": lat,
					"lon": lon,
					"postal_code": postal_code,
					"label": label,
					"split": split_name,
					"heading": heading,
					"pitch": pitch,
					"fov": fov,
					"size": size,
					"pano_id": pano_id,
					"capture_date": capture_date,
					"status": "OK",
					"path": str(out_path),
					"error": "",
				}
			except Exception as exc:
				counters["image_failed"] += 1
				row_data = {
					"id": sample_id,
					"lat": lat,
					"lon": lon,
					"postal_code": postal_code,
					"label": label,
					"split": split_name,
					"heading": heading,
					"pitch": pitch,
					"fov": fov,
					"size": size,
					"pano_id": pano_id,
					"capture_date": capture_date,
					"status": "DOWNLOAD_FAILED",
					"path": str(out_path),
					"error": str(exc),
				}
				logging.warning("%s | heading=%s failed: %s", sample_id, heading, exc)

			write_metadata_row(combined_writer, row_data)
			write_metadata_row(split_writer, row_data)
			time.sleep(min_delay_seconds)

	return counters

'''
--------------------------------
ensure_output_dirs(output_root)
--------------------------------
Description:
Ensures that the necessary output directories for the dataset splits
(train, validation, test) exist within the specified output root directory.
Creates the directories if they do not already exist, and returns the paths to
these directories for use in saving downloaded images for each split.

Arguments:
output_root: The root directory under which the split directories should be created.

Returns:
A tuple of three Path objects corresponding to the train, validation, and test directories.
'''
def ensure_output_dirs(output_root):
	train_dir = output_root / "Street_Training_Set"
	valid_dir = output_root / "Street_Validation_Set"
	test_dir = output_root / "Street_Test_Set"

	for directory in (train_dir, valid_dir, test_dir):
		directory.mkdir(parents=True, exist_ok=True)

	return train_dir, valid_dir, test_dir

'''
--------------------------------
run()
--------------------------------
Description:
The main execution function for the dataset extraction script. It sets 
up logging, parses and validates command-line arguments, loads samples
from the input CSV, splits them into train/validation/test sets, and
processes each split by querying Street View metadata and downloading 
images. It maintains and logs statistics about the number of samples
processed, metadata successes/failures, and image download successes/failures.
Finally, it logs the locations of the combined metadata CSV and the per-split
metadata CSV files for reference.

Arguments:
None. The function relies on command-line arguments for configuration.

Returns:
None. The function performs its operations and logs results to the console and CSV files.
'''
def run():
	setup_logging()
	args = parse_args()
	validate_args(args)

	headings = parse_headings(args.headings)
	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	train_dir, valid_dir, test_dir = ensure_output_dirs(output_root)

	combined_metadata_path = output_root / args.metadata_csv
	split_metadata_paths = {
		"train": output_root / "street_metadata_train.csv",
		"valid": output_root / "street_metadata_valid.csv",
		"test": output_root / "street_metadata_test.csv",
	}

	logging.info("Loading samples from %s", args.input_csv)
	with requests.Session() as session:
		samples = load_samples(
			args.input_csv,
			session,
			timeout=args.timeout_seconds,
			max_retries=args.max_retries,
			backoff_seconds=args.retry_backoff_seconds,
			geocode_delay_seconds=args.geocode_delay_seconds,
		)

		train_rows, valid_rows, test_rows = split_rows(
			samples,
			seed=args.seed,
			train_ratio=args.train_ratio,
			valid_ratio=args.valid_ratio,
		)

		logging.info(
			"Loaded %s samples | train=%s valid=%s test=%s",
			len(samples),
			len(train_rows),
			len(valid_rows),
			len(test_rows),
		)

		with open(combined_metadata_path, "w", newline="", encoding="utf-8") as combined_file, \
			open(split_metadata_paths["train"], "w", newline="", encoding="utf-8") as train_meta_file, \
			open(split_metadata_paths["valid"], "w", newline="", encoding="utf-8") as valid_meta_file, \
			open(split_metadata_paths["test"], "w", newline="", encoding="utf-8") as test_meta_file:

			combined_writer = csv.writer(combined_file)
			train_writer = csv.writer(train_meta_file)
			valid_writer = csv.writer(valid_meta_file)
			test_writer = csv.writer(test_meta_file)

			combined_writer.writerow(METADATA_COLUMNS)
			train_writer.writerow(METADATA_COLUMNS)
			valid_writer.writerow(METADATA_COLUMNS)
			test_writer.writerow(METADATA_COLUMNS)

			total = {
				"samples": 0,
				"metadata_ok": 0,
				"metadata_not_ok": 0,
				"image_ok": 0,
				"image_failed": 0,
			}

			split_tasks = [
				(train_rows, train_dir, "train", train_writer),
				(valid_rows, valid_dir, "valid", valid_writer),
				(test_rows, test_dir, "test", test_writer),
			]

			for split_rows_data, split_dir, split_name, split_writer in split_tasks:
				logging.info("Exporting split: %s", split_name)
				stats = export_split(
					split_rows_data,
					split_dir,
					split_name,
					session=session,
					api_key=args.api_key,
					headings=headings,
					pitch=args.pitch,
					fov=args.fov,
					size=args.size,
					radius=args.radius,
					source=args.source,
					timeout=args.timeout_seconds,
					max_retries=args.max_retries,
					backoff_seconds=args.retry_backoff_seconds,
					min_delay_seconds=args.min_delay_seconds,
					metadata_only=args.metadata_only,
					combined_writer=combined_writer,
					split_writer=split_writer,
				)

				for key in total:
					total[key] += stats[key]

				logging.info(
					"Split %s complete | samples=%s metadata_ok=%s metadata_not_ok=%s image_ok=%s image_failed=%s",
					split_name,
					stats["samples"],
					stats["metadata_ok"],
					stats["metadata_not_ok"],
					stats["image_ok"],
					stats["image_failed"],
				)

	logging.info("All splits complete.")
	logging.info(
		"TOTAL | samples=%s metadata_ok=%s metadata_not_ok=%s image_ok=%s image_failed=%s",
		total["samples"],
		total["metadata_ok"],
		total["metadata_not_ok"],
		total["image_ok"],
		total["image_failed"],
	)
	logging.info("Combined metadata: %s", combined_metadata_path)
	logging.info("Per-split metadata: %s", split_metadata_paths)

# main entry point
if __name__ == "__main__":
	run()
