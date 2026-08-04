"""
Dataset extraction script for Google Maps Tile images.

Author: AxeWhy
Date: 2026-04-15
Last updated: 2026-04-20

Description:
Builds a Google Maps Tile image dataset from a CSV of locations and labels.
Supports both coordinate and postal-code input formats, performs
train/validation/test splitting, downloads images using the Satellite Tile API 
at multiple zoom levels, and writes detailed metadata logs.

Input CSV formats supported:
1) id,lat,lon,label
2) id,postal_code,label

Usage example:
python Dataset/Extraction_Map.py --input-csv Dataset/hdb_points_postal.csv --api-key <YOUR_API_KEY>

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

'''
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
Description:
Parses command-line arguments for the dataset extraction script. Supports options for
input/output paths, dataset splitting ratios, Google Maps Tile API parameters, and 
request retry settings. Validates required API key presence and argument value ranges.

Arguments:
--input-csv: Path to input CSV file with location data (default: Dataset/Input/hdb_points_postal.csv)
--output-root: Root directory for output dataset (default: Dataset/Map)
--metadata-csv: Filename for combined metadata CSV (default: map_metadata.csv)
--api-key: Google Maps API key (default: from GOOGLE_MAPS_API_KEY env var)
--seed: Random seed for dataset splitting (default: 42)
--train-ratio: Proportion of samples for training set (default: 0.7
--valid-ratio: Proportion of samples for validation set (default: 0.15)
--map-type: Google Map Tiles map type (satellite, roadmap, terrain; default: satellite)
--zoom-levels: Comma-separated zoom levels to download (default: 20)

Returns:
Namespace with parsed arguments.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
Description:
Configures logging to output timestamped messages to the console. Uses INFO level by default.

Arguments:
None

Returns:
None
'''
def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

'''
# ----------------------------------------------------------------------------------------------------
# validate_args(args)
# ----------------------------------------------------------------------------------------------------
Description:
Validates parsed command-line arguments for the dataset extraction script. Checks for required
API key presence, ensures split ratios are valid, verifies zoom levels are within allowed range,
and confirms map type and image format are supported. Raises ValueError with descriptive messages
for any invalid arguments.

Arguments:
args: Namespace of parsed arguments from parse_args()

Returns:
None (raises ValueError on invalid arguments)
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# parse_zoom_levels(zoom_levels_text)
# ----------------------------------------------------------------------------------------------------
Description:
Parses a comma-separated string of zoom levels into a list of unique integers. Validates that
each zoom level is within the allowed range [0, MAX_TILE_ZOOM_LEVEL]. Raises ValueError if any
zoom level is invalid or if no valid levels are provided. Preserves the order of zoom levels 
as specified in the input string while deduplicating.

Arguments:
zoom_levels_text: Comma-separated string of zoom levels, e.g. "19,20,21"

Returns:
List of unique zoom levels as integers, e.g. [19, 20, 21]
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# _row_value(row, *keys)
# ----------------------------------------------------------------------------------------------------
Description:
Helper function to extract the first non-empty value from a dictionary row for a 
list of possible keys. Used to handle variations in input CSV column names 
(e.g. "lat" vs "latitude"). Returns the first non-empty string value found, or 
None if all keys are missing or empty. Strips whitespace from values and converts them
to string before checking for emptiness.

Arguments:
row: Dictionary representing a CSV row
*keys: Variable list of keys to check in the row, in order of preference

Returns:
First non-empty string value found for the provided keys, or None if all are missing/empty.
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
# ----------------------------------------------------------------------------------------------------
# sanitize_filename(value)
# ----------------------------------------------------------------------------------------------------
Description:
Sanitizes a string to be safe for use as a filename by replacing any character that is not
a letter, digit, dot, underscore, or hyphen with an underscore. This helps prevent issues with
invalid characters in filenames when saving downloaded tiles. Converts the input to string before
processing to handle non-string values gracefully. Does not perform any truncation or length checks,
so very long values may still produce long filenames after sanitization.

Arguments:
value: The input value to sanitize, typically a sample ID or similar identifier.

Returns:
Sanitized string safe for use as a filename.
'''
def sanitize_filename(value):
	return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))

'''
# ----------------------------------------------------------------------------------------------------
# normalize_image_format(image_format)
# ----------------------------------------------------------------------------------------------------
Description:
Normalizes an image format string by stripping whitespace, converting to lowercase, and
removing any "image/" prefix. Also converts "jpg" to "jpeg" for consistency. This helps ensure
that different ways of specifying image formats (e.g. "image/jpeg", "JPEG", "jpg") are treated
consistently when determining file extensions and validating supported formats. Returns the
normalized format string, or an empty string if the input is None or empty after stripping.

Arguments:
image_format: The input image format string to normalize, e.g. "image/jpeg", "JPEG", "jpg"

Returns:
Normalized image format string, e.g. "jpeg", or an empty string if input is None/empty.
'''
def normalize_image_format(image_format):
	text = str(image_format or "").strip().lower()
	if text.startswith("image/"):
		text = text.split("/", 1)[1]
	if text == "jpg":
		text = "jpeg"
	return text

'''
# ----------------------------------------------------------------------------------------------------
# image_format_to_extension(image_format)
# ----------------------------------------------------------------------------------------------------
Description:
Maps a normalized image format string to a common file extension. For "jpeg", returns "jpg".
For "png" and "webp", returns the same string. For any unsupported or unknown format, returns "bin"
as a generic binary extension. This function is used to determine the file extension to use when
saving downloaded tiles based on the image format specified in the API response or preferred by the
user. The input image_format should already be normalized using normalize_image_format() for consistent
results.

Arguments:
image_format: Normalized image format string, e.g. "jpeg", "png", "webp"

Returns:
File extension string to use for saving the tile, e.g. "jpg" for "jpeg", or "bin" for unknown formats.
'''
def image_format_to_extension(image_format):
	fmt = normalize_image_format(image_format)
	if fmt == "jpeg":
		return "jpg"
	if fmt in {"png", "webp"}:
		return fmt
	return "bin"

'''
# ----------------------------------------------------------------------------------------------------
# parse_iso_datetime(text)
# ----------------------------------------------------------------------------------------------------
Description:
Parses an ISO 8601 datetime string into a timezone-aware datetime object. Handles strings that end
with "Z" by treating them as UTC. If the parsed datetime does not include timezone information, it
is assumed to be in UTC. Returns None if the input text is None, empty, or cannot be parsed as a valid
ISO datetime. This function is used to parse the "expiry" field from the tile session response, which
is expected to be in the ISO 8601 format. The resulting datetime object will be timezone-aware and in UTC,
which allows for accurate comparisons with the current time when determining if the session needs to be
refreshed.

Arguments:
text: The input ISO 8601 datetime string to parse, e.g. "2026-04-15T12:34:56Z"

Returns:
Timezone-aware datetime object in UTC if parsing is successful, or None if input is invalid.
'''
def parse_iso_datetime(text):
	if not text:
		return None
	raw = str(text).strip()
	if raw.endswith("Z"):
		raw = raw[:-1] + "+00:00"
	try:
		dt_obj = datetime.datetime.fromisoformat(raw)
	except ValueError:
		return None
	if dt_obj.tzinfo is None:
		dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
	return dt_obj

'''
# ----------------------------------------------------------------------------------------------------
# request_with_retries(session, method, url, *, params=None, json_payload=None,
#                      timeout=30, max_retries=4, backoff_seconds=1.2)
# ----------------------------------------------------------------------------------------------------
Description:
Performs an HTTP request with retries and exponential backoff for transient errors. Retries are
triggered for HTTP status codes in RETRY_HTTP_STATUSES and for any RequestException. The function 
will attempt the request up to max_retries times, with a delay that increases exponentially based on
the attempt number. If all attempts fail, a RuntimeError is raised with the last encountered error as 
the cause. This function is used for all HTTP interactions with the Google Map Tile API and the OneMap
geocoding API to improve robustness against transient network issues and rate limiting.

Arguments:
session: requests.Session object to use for the request
method: HTTP method as a string, e.g. "GET", "POST"
url: The URL to send the request to
params: Optional dictionary of query parameters to include in the request
json_payload: Optional dictionary to send as JSON body in the request
timeout: Timeout in seconds for the request (default: 30)
max_retries: Maximum number of retry attempts (default: 4)
backoff_seconds: Base number of seconds to use for exponential backoff (default: 1.2)

Returns:
requests.Response object if the request is successful

Raises:
RuntimeError if all retry attempts fail, with the last encountered error as the cause.
'''
def request_with_retries(
	session,
	method,
	url,
	*,
	params=None,
	json_payload=None,
	timeout=30,
	max_retries=4,
	backoff_seconds=1.2,
):
	last_error = None
	for attempt in range(max_retries + 1):
		try:
			response = session.request(
				method,
				url,
				params=params,
				json=json_payload,
				timeout=timeout,
			)

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
# ----------------------------------------------------------------------------------------------------
# geocode_postal_sg(session, postal_code, timeout, max_retries, backoff_seconds)
# ----------------------------------------------------------------------------------------------------
Description:
Geocodes a Singapore postal code to latitude and longitude using the OneMap API. Returns a tuple
of (latitude, longitude) if successful, or (None, None) if the postal code is not found or if there
is an error during the request. Uses the request_with_retries function to perform the API call with
retries and backoff. The function checks the API response for the "found" field to determine if any
result were found, and extracts the latitude and longitude from the first result if available. This 
function is used to convert postal codes to coordinates when the input CSV does not provide lat/lon
values directly, allowing for more flexible input formats while still supporting geocoding for
Singapore locations.

Arguments:
session: requests.Session object to use for the API request
postal_code: The postal code to geocode as a string
timeout: Timeout in seconds for the API request
max_retries: Maximum number of retry attempts for the API request
backoff_seconds: Base number of seconds to use for exponential backoff between retries

Returns:
Tuple of (latitude, longitude) as floats if geocoding is successful
(None, None) if not found or on error.
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
# ----------------------------------------------------------------------------------------------------
# load_samples(csv_path, session, timeout, max_retries, backoff_seconds, geocode_delay_seconds)
# ----------------------------------------------------------------------------------------------------
Description:
Loads samples from the input CSV file and returns a list of dictionaries with keys:
id, lat, lon, label, postal_code. The function supports CSVs with either lat/lon columns or postal
code columns. For rows with postal codes, it uses the geocode_postal_sg function to obtain lat/lon
coordinates. The function performs validation on the loaded samples, ensuring that lat/lon values are
within valid ranges for Web Mercator tiles. Invalid rows are skipped with a warning logged. The function
also cache geocoding results to avoid redundant API calls for the same postal code. If no valid samples 
are loaded, a ValueError is raised. This function is responsible for the initial data loading and 
preprocessing step, converting the input CSV into a standardized format with coordinates ready for
tile extraction.

Arguments:
csv_path: Path to the input CSV file
session: requests.Session object to use for geocoding API calls
timeout: Timeout in seconds for API requests
max_retries: Maximum number of retry attempts for API requests
backoff_seconds: Base number of seconds to use for exponential backoff between retries
geocode_delay_seconds: Number of seconds to wait between geocoding API calls to avoid rate limits

Returns:
List of dictionaries representing valid samples with keys: id, lat, lon, label, postal_code

Raises:
ValueError if no valid samples are loaded from the CSV.
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

			if not (-85.051129 <= lat <= 85.051129):
				logging.warning(
					"Skipping line %s (%s): latitude out of Web Mercator range.",
					line_no,
					sample_id,
				)
				continue
			if not (-180 <= lon <= 180):
				logging.warning(
					"Skipping line %s (%s): longitude out of range.",
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
# ----------------------------------------------------------------------------------------------------
# split_rows(rows, seed, train_ratio, valid_ratio)
# ----------------------------------------------------------------------------------------------------
Description:
Splits a list of rows into train, validation, and test sets based on the provided ratios. The
function shuffles the input rows using a random.Random instance initialized with the given seed
for reproducibility. The number of samples in each split is determined by the train_ratio and valid_ratio,
with the remaining samples assigned to the test set. The function returns three lists of rows 
corresponding to the train, validation, and test splits. This function is used to perform the dataset 
splitting after loading the samples, ensuring that the splits are randomized but reproducible based on
the provided seed. The split ratios are validated in the validate_args function to ensure they are within
acceptable ranges before this function is called.

Arguments:
rows: List of dictionaries representing the samples to split
seed: Random seed for shuffling the rows to ensure reproducibility
train_ratio: Proportion of samples to assign to the training set (e.g. 0.7)
valid_ratio: Proportion of samples to assign to the validation set (e.g. 0.15)

Returns:
Tuple of three lists: (train_rows, valid_rows, test_rows) corresponding to the splits.
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
# ----------------------------------------------------------------------------------------------------
# lat_lon_to_tile_xy(lat, lon, zoom)
# ----------------------------------------------------------------------------------------------------
Description:
Converts latitude and longitude coordinates to Google Map Tile X and Y tile coordinates for a given
zoom level. The function first converts the latitude to radians and calculates the number of tiles
at the specified zoom level. It then computes the floating-point tile coordinates based on the Web
Mercator projection formulas. Finally, it converts the floating-point tile coordinates to integers
and clamps them to the valid range of [0, n-1] where n is the number of tiles at that zoom level.
This function is used to determine which tile to download for a given location and zoom level
when extracting the dataset. The resulting tile_x and tile_y coordinate values can then be used to
construct the URL for the tile image and to organize the downloaded files in the output dataset structure.

Arguments:
lat: Latitude of the location in degrees
lon: Longitude of the location in degrees
zoom: Zoom level for the tile (non-negative integer)

Returns:
Tuple of (tile_x, tile_y) as integers representing the tile coordinates at the specified zoom level.
'''
def lat_lon_to_tile_xy(lat, lon, zoom):
	lat_rad = math.radians(lat)
	n = 2 ** zoom

	x_float = (lon + 180.0) / 360.0 * n
	y_float = (
		(1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0
	) * n

	tile_x = int(x_float)
	tile_y = int(y_float)

	tile_x = min(max(tile_x, 0), n - 1)
	tile_y = min(max(tile_y, 0), n - 1)
	return tile_x, tile_y

'''
# ----------------------------------------------------------------------------------------------------
# extract_status_code(exc)
# ----------------------------------------------------------------------------------------------------
Description:
Helper function to extract the HTTP status code from a RuntimeError that was caused by a requests.HTTPError.
This is used to determine if a tile download failure was due to a specific HTTP error that should trigger a 
session refresh. The function checks if the cause of the RuntimeError is an HTTPError and if it has a response
object, then returns the status code from that response. If the cause is not an HTTPError or does not have a 
response, the function returns None. This allows the calling code to check the status code of the failure and
decide whether to refresh the tile session and retry the download, or to treat it as a non-recoverable error.

Arguments:
exc: The RuntimeError exception from which to extract the HTTP status code, if available.

Returns:
The HTTP status code as an integer if the cause of the RuntimeError is a requests.HTTPError with a response,
or None otherwise.
'''
def extract_status_code(exc):
	cause = exc.__cause__
	if isinstance(cause, requests.HTTPError) and cause.response is not None:
		return cause.response.status_code
	return None

'''
# ----------------------------------------------------------------------------------------------------
# safe_write_bytes(out_path, data)
# ----------------------------------------------------------------------------------------------------
Description:
Writes binary data to a file in a safe manner by first writing to a temporary file and then 
atomically renaming it to the target path. This helps prevent issues with partial writes or
corrupted files if the process is interrupted during the write operation. The function takes
a path object for the output file and the binary data to write. It creates a temporary file
with the same name as the target file but with an additional ".tmp" suffix, writes the data
to that temporary file, and then replace the temporary file with the target file using an atomic
operation. This is used when saving downloaded tile images to ensure that the output files are 
either fully written or not created at all, avoiding the possibility of ending up with incomplete
or corrupted tile files in the dataset if something goes wrong during the download or write process.

Arguments:
out_path: Path object representing the target file path to write to
data: Binary data to write to the file

Returns:
None (raises exceptions on failure)
'''
def safe_write_bytes(out_path, data):
	temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
	temp_path.write_bytes(data)
	temp_path.replace(out_path)

'''
# ----------------------------------------------------------------------------------------------------
# MapTilesClient class
# ----------------------------------------------------------------------------------------------------
Description:
Client class to manage interactions with the Google Map Tiles API, including session 
management and tile downloading. The class handles creating and refreshing tile sessions,
constructing tile URLs, and downloading tiles with retries. It also keeps track of the 
current session ID, expiry time, tile dimensions, and image format based on the API responses.

Member functions:
__init__: 
Initializes the MapTilesClient with the provided parameters and sets up the initial state.

create_session:
The create_session function initializes or refreshes the tile session by making a POST request 
to the API with the specified parameters. 

ensure_session:
The ensure_session function checks if the session needs to be refreshed based on the expiry 
time and calls create_session if necessary. 

download_tile:
The download_tile function attempts to download a tile for the given zoom level and tile coordinates, and if it
encounters an HTTP error that indicates the session has expired, it will refresh the session and
retry the download once. 

file_extension:
The file_extension function returns the appropriate file extension to
use when saving tiles based on the image format specified in the session. This class encapsulates all
the logic related to managing the tile session and downloading tiles, providing a clean interface
for the main dataset extraction logic to use when it needs to download tiles for the samples.
'''
class MapTilesClient:
	def __init__(
		self,
		session,
		*,
		api_key,
		map_type,
		language,
		region,
		preferred_image_format,
		timeout,
		max_retries,
		backoff_seconds,
		refresh_seconds,
	):
		self.http_session = session
		self.api_key = api_key
		self.map_type = map_type
		self.language = language
		self.region = region
		self.preferred_image_format = normalize_image_format(preferred_image_format)
		self.timeout = timeout
		self.max_retries = max_retries
		self.backoff_seconds = backoff_seconds
		self.refresh_seconds = refresh_seconds

		self.tiles_session_id = None
		self.expiry = None
		self.tile_width = ""
		self.tile_height = ""
		self.image_format = self.preferred_image_format

	def _needs_refresh(self):
		if not self.tiles_session_id:
			return True
		if self.expiry is None:
			return False

		now = datetime.datetime.now(datetime.timezone.utc)
		refresh_at = now + datetime.timedelta(seconds=self.refresh_seconds)
		return refresh_at >= self.expiry

	def create_session(self, force=False):
		if not force and not self._needs_refresh():
			return

		payload = {
			"mapType": self.map_type,
			"language": self.language,
			"region": self.region,
			"imageFormat": self.preferred_image_format,
		}

		response = request_with_retries(
			self.http_session,
			"POST",
			CREATE_SESSION_URL,
			params={"key": self.api_key},
			json_payload=payload,
			timeout=self.timeout,
			max_retries=self.max_retries,
			backoff_seconds=self.backoff_seconds,
		)
		body = response.json()

		session_id = body.get("session")
		if not session_id:
			raise RuntimeError("Google Map Tiles session response did not include session ID.")

		self.tiles_session_id = session_id
		self.expiry = parse_iso_datetime(body.get("expiry"))
		self.tile_width = body.get("tileWidth", "")
		self.tile_height = body.get("tileHeight", "")
		self.image_format = normalize_image_format(
			body.get("imageFormat", self.preferred_image_format)
		)

	def ensure_session(self):
		if self._needs_refresh():
			self.create_session(force=True)

	def file_extension(self):
		return image_format_to_extension(self.image_format)

	def download_tile(self, zoom, tile_x, tile_y, out_path):
		for attempt in range(2):
			self.ensure_session()

			url = TILE_URL_TEMPLATE.format(z=zoom, x=tile_x, y=tile_y)
			params = {
				"session": self.tiles_session_id,
				"key": self.api_key,
			}

			try:
				response = request_with_retries(
					self.http_session,
					"GET",
					url,
					params=params,
					timeout=self.timeout,
					max_retries=self.max_retries,
					backoff_seconds=self.backoff_seconds,
				)
				safe_write_bytes(out_path, response.content)
				return
			except RuntimeError as exc:
				status_code = extract_status_code(exc)
				if status_code in SESSION_REFRESH_HTTP_STATUSES and attempt == 0:
					logging.info(
						"Refreshing tile session after HTTP %s for z=%s x=%s y=%s.",
						status_code,
						zoom,
						tile_x,
						tile_y,
					)
					self.create_session(force=True)
					continue
				raise

		raise RuntimeError(
			f"Tile download failed after session refresh: z={zoom}, x={tile_x}, y={tile_y}."
		)

'''
# ----------------------------------------------------------------------------------------------------
# write_metadata_row(writer, row)
# ----------------------------------------------------------------------------------------------------
Description:
Writes a row of metadata to the given CSV writer object. The row is expected to be a dictionary
with keys corresponding to the columns defined in METADATA_COLUMNS. The function extracts the
values for each column from the row dictionary, using an empty string if a key is missing, and writes
them as a list to the CSV writer. This function is used to write metadata about each sample and tile
to the combined metadata CSV and the split-specific metadata CSVs during the export process. It ensures
that the metadata rows are consistently formatted according to the defined columns, even if some values
are missing from the input row dictionary.

Arguments:
writer: csv.writer object to write the row to
row: Dictionary containing the metadata values for the sample and tile, 
with keys corresponding to METADATA_COLUMNS

Returns:
None (writes the row to the CSV writer)
'''
def write_metadata_row(writer, row):
	writer.writerow([row.get(col, "") for col in METADATA_COLUMNS])

'''
# ----------------------------------------------------------------------------------------------------
# export_split(rows, split_dir, split_name, *, map_type, zoom_levels, tile_client,
#              min_delay_seconds, metadata_only, skip_existing, combined_writer, split_writer)
# ----------------------------------------------------------------------------------------------------
Description:
Exports a list of rows for a specific dataset split (train/valid/test) by downloading the corresponding
map tiles and writing metadata about each sample and tile to the provided CSV writers. For each row,
the function iterates over the specified zoom levels, calculates the tile coordinates, and attempts to
download the tile using the MapTilesClient class. It handles various conditions such as skipping existing
tiles, only writing metadata without downloading, and logging any download failures. The function keeps
a counter of the number of sample processed, tile targets, successful downloads, failed downloads, and
skipped tiles, which it returns at the end. This function encapsulates the core logic of exporting the
dataset for a given split, including the interaction with the tile client and the writing of metadata.

Arguments:
rows: List of dictionaries representing the samples to export for this split
split_dir: Path object representing the directory to save the downloaded tiles for this split
split_name: Name of the split (e.g. "train", "valid", "test") for labeling and logging purposes
map_type: The map type to use for downloading tiles (e.g. "satellite")
zoom_levels: List of zoom levels to download for each sample
tile_client: An instance of MapTilesClient to use for downloading tiles
min_delay_seconds: Minimum number of seconds to wait between processing each tile to avoid rate limits
metadata_only: If True, only write metadata rows without downloading tiles
skip_existing: If True, skip downloading tiles that already exist at the target path
combined_writer: csv.writer object for writing to the combined metadata CSV
split_writer: csv.writer object for writing to the split-specific metadata CSV

Returns:
Dictionary of counters with keys: "samples", "tile_targets", "tile_ok", "tile_failed", "tile_skipped"
'''
def export_split(
	rows,
	split_dir,
	split_name,
	*,
	map_type,
	zoom_levels,
	tile_client,
	min_delay_seconds,
	metadata_only,
	skip_existing,
	combined_writer,
	split_writer,
):
	counters = {
		"samples": len(rows),
		"tile_targets": len(rows) * len(zoom_levels),
		"tile_ok": 0,
		"tile_failed": 0,
		"tile_skipped": 0,
	}

	for row in rows:
		sample_id = row["id"]
		lat = row["lat"]
		lon = row["lon"]
		label = row["label"]
		postal_code = row.get("postal_code", "")
		safe_id = sanitize_filename(sample_id)

		for zoom in zoom_levels:
			tile_x, tile_y = lat_lon_to_tile_xy(lat, lon, zoom)
			extension = tile_client.file_extension()
			file_name = f"{safe_id}_{split_name}_z{zoom}_x{tile_x}_y{tile_y}_{postal_code}.{extension}"
			out_path = split_dir / file_name

			row_data = {
				"id": sample_id,
				"lat": lat,
				"lon": lon,
				"postal_code": postal_code,
				"label": label,
				"split": split_name,
				"map_type": map_type,
				"zoom": zoom,
				"tile_x": tile_x,
				"tile_y": tile_y,
				"tile_width": tile_client.tile_width,
				"tile_height": tile_client.tile_height,
				"image_format": tile_client.image_format,
				"session_id": tile_client.tiles_session_id,
				"status": "",
				"path": str(out_path),
				"error": "",
			}

			if metadata_only:
				row_data["status"] = "OK_METADATA_ONLY"
				write_metadata_row(combined_writer, row_data)
				write_metadata_row(split_writer, row_data)
				time.sleep(min_delay_seconds)
				continue

			if skip_existing and out_path.exists():
				counters["tile_skipped"] += 1
				row_data["status"] = "SKIPPED_EXISTS"
				write_metadata_row(combined_writer, row_data)
				write_metadata_row(split_writer, row_data)
				time.sleep(min_delay_seconds)
				continue

			try:
				tile_client.download_tile(zoom, tile_x, tile_y, out_path)
				counters["tile_ok"] += 1
				row_data["tile_width"] = tile_client.tile_width
				row_data["tile_height"] = tile_client.tile_height
				row_data["image_format"] = tile_client.image_format
				row_data["session_id"] = tile_client.tiles_session_id
				row_data["status"] = "OK"
			except Exception as exc:
				counters["tile_failed"] += 1
				row_data["status"] = "DOWNLOAD_FAILED"
				row_data["error"] = str(exc)
				logging.warning(
					"%s | split=%s zoom=%s tile=%s,%s failed: %s",
					sample_id,
					split_name,
					zoom,
					tile_x,
					tile_y,
					exc,
				)

			write_metadata_row(combined_writer, row_data)
			write_metadata_row(split_writer, row_data)
			time.sleep(min_delay_seconds)

	return counters

'''
# ----------------------------------------------------------------------------------------------------
# ensure_output_dirs(output_root)
# ----------------------------------------------------------------------------------------------------
Description:
Ensures that the output directories for the train, validation, and test splits exist under the given
output root directory. The function constructs the paths for the "Map_Training_Set", "Map_Validation_Set",
and "Map_Test_Set" directories, creates them if they do not already exist, and returns the Path objects
for each of the split directories. This function is used at the beginning of the run() function to set
up the directory structure for saving the downloaded tiles for each split of the dataset. By ensuring
the directories exist before starting the export process, it prevents errors related to missing 
directories when the code attempts to save tile images during the export_split function.

Arguments:
output_root: Path object representing the root directory under which the split directories will be created

Returns:
Tuple of Path objects: (train_dir, valid_dir, test_dir) representing the directories for each split
'''
def ensure_output_dirs(output_root):
	train_dir = output_root / "Map_Training_Set"
	valid_dir = output_root / "Map_Validation_Set"
	test_dir = output_root / "Map_Test_Set"

	for directory in (train_dir, valid_dir, test_dir):
		directory.mkdir(parents=True, exist_ok=True)

	return train_dir, valid_dir, test_dir

'''
# ----------------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------------
Description:
Main execution function for the dataset extraction process. It sets up logging, parses and validates
command-line arguments, loads samples from the input CSV, splits them into train/valid/test sets, and
iterates over each split to export the tiles and metadata. The function uses a requests.Session for 
HTTP interactions and initializes a MapTilesClient class for mapping API interactions. It writes metadata
to both a combined metadata CSV and split-specific metadata CSVs. The function also keeps track of overall
statistics for the number of samples processed, tile targets, successful downloads, failed downloads, and
skipped tiles, which it logs at the end of the process. This function orchestrates the entire workflow of
the dataset extraction, coordinating the loading of data, interaction with the tile API, and saving of
results.

Arguments:
None (all configuration is done through command-line arguments)

Returns:
None (the function performs the dataset extraction and saves results to disk)
'''
def run():
	setup_logging()
	args = parse_args()
	validate_args(args)

	zoom_levels = parse_zoom_levels(args.zoom_levels)
	map_type = str(args.map_type).strip().lower()

	output_root = Path(args.output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	train_dir, valid_dir, test_dir = ensure_output_dirs(output_root)

	combined_metadata_path = output_root / args.metadata_csv
	split_metadata_paths = {
		"train": output_root / "map_metadata_train.csv",
		"valid": output_root / "map_metadata_valid.csv",
		"test": output_root / "map_metadata_test.csv",
	}

	logging.info("Loading samples from %s", args.input_csv)
	with requests.Session() as session:
		session.headers.update({"User-Agent": "NEA-MapTileExtractor/1.0"})

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
			"Loaded %s samples | train=%s valid=%s test=%s | map_type=%s zoom_levels=%s",
			len(samples),
			len(train_rows),
			len(valid_rows),
			len(test_rows),
			map_type,
			zoom_levels,
		)

		tile_client = MapTilesClient(
			session,
			api_key=args.api_key,
			map_type=map_type,
			language=args.language,
			region=args.region,
			preferred_image_format=args.image_format,
			timeout=args.timeout_seconds,
			max_retries=args.max_retries,
			backoff_seconds=args.retry_backoff_seconds,
			refresh_seconds=args.session_refresh_seconds,
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
				"tile_targets": 0,
				"tile_ok": 0,
				"tile_failed": 0,
				"tile_skipped": 0,
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
					map_type=map_type,
					zoom_levels=zoom_levels,
					tile_client=tile_client,
					min_delay_seconds=args.min_delay_seconds,
					metadata_only=args.metadata_only,
					skip_existing=args.skip_existing,
					combined_writer=combined_writer,
					split_writer=split_writer,
				)

				for key in total:
					total[key] += stats[key]

				logging.info(
					"Split %s complete | samples=%s tile_targets=%s tile_ok=%s tile_failed=%s tile_skipped=%s",
					split_name,
					stats["samples"],
					stats["tile_targets"],
					stats["tile_ok"],
					stats["tile_failed"],
					stats["tile_skipped"],
				)

	logging.info("All splits complete.")
	logging.info(
		"TOTAL | samples=%s tile_targets=%s tile_ok=%s tile_failed=%s tile_skipped=%s",
		total["samples"],
		total["tile_targets"],
		total["tile_ok"],
		total["tile_failed"],
		total["tile_skipped"],
	)
	logging.info("Combined metadata: %s", combined_metadata_path)
	logging.info("Per-split metadata: %s", split_metadata_paths)

# Main entry point
if __name__ == "__main__":
	run()
