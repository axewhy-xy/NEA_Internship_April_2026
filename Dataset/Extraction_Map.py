"""
Dataset extraction script for Google Maps Tile images.

Author: AxeWhy
Date: 2026-04-15
Last updated: 2026-04-15

Description:
Builds a Google Maps Tile image dataset from a CSV of locations and labels.
Supports both coordinate and postal-code input formats, performs
train/validation/test splitting, downloads images using the Satellite Tile API 
at multiple zoom levels, and writes detailed metadata logs.

Input CSV formats supported:
1) id,lat,lon,label
2) id,postal_code,label

Usage example:
python Dataset/Extraction_Map.py --input-csv Dataset/hdb_points_postal.csv

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

def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)

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

def _row_value(row, *keys):
	for key in keys:
		value = row.get(key)
		if value is None:
			continue
		text = str(value).strip()
		if text:
			return text
	return None

def normalize_image_format(image_format):
	text = str(image_format or "").strip().lower()
	if text.startswith("image/"):
		text = text.split("/", 1)[1]
	if text == "jpg":
		text = "jpeg"
	return text

def ensure_output_dirs(output_root):
	train_dir = output_root / "Map_Training_Set"
	valid_dir = output_root / "Map_Validation_Set"
	test_dir = output_root / "Map_Test_Set"

	for directory in (train_dir, valid_dir, test_dir):
		directory.mkdir(parents=True, exist_ok=True)

	return train_dir, valid_dir, test_dir

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

def image_format_to_extension(image_format):
	fmt = normalize_image_format(image_format)
	if fmt == "jpeg":
		return "jpg"
	if fmt in {"png", "webp"}:
		return fmt
	return "bin"

def safe_write_bytes(out_path, data):
	temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
	temp_path.write_bytes(data)
	temp_path.replace(out_path)
	
def extract_status_code(exc):
	cause = exc.__cause__
	if isinstance(cause, requests.HTTPError) and cause.response is not None:
		return cause.response.status_code
	return None

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

def sanitize_filename(value):
	return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))

def write_metadata_row(writer, row):
	writer.writerow([row.get(col, "") for col in METADATA_COLUMNS])

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
			file_name = f"{safe_id}_{split_name}_z{zoom}_x{tile_x}_y{tile_y}.{extension}"
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


if __name__ == "__main__":
	run()