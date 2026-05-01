"""WB-LEN ingestion: STAC catalog walking and windowed COG reads.

This module replaces the old `downloader.py`. Instead of downloading whole-globe
annual composites and then cropping them locally, it:

  1. Walks the static STAC catalog at s3://globalnightlight/ to enumerate the
     per-orbit nightly granules that intersect a region of interest.
  2. For each orbit, resolves the three layers we care about (radiance, lunar
     illuminance, QA flag) and reads only the ROI window of each COG, caching
     the small clipped raster locally.

Step 1 deliverable. Compositing, masking, and harmonization happen downstream.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tqdm import tqdm

from harmonizer.constants import (
    S3_BUCKET,
    S3_HTTPS_BASE,
    SENSOR_CONFIGS,
    SENSOR_DMSP,
    SENSOR_VIIRS,
    SensorConfig,
    viirs_orbit_key,
)

log = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in EPSG:4326
S3_LIST_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _list_s3_keys(prefix: str, bucket: str = S3_BUCKET) -> list[str]:
    """List object keys under a prefix in an anonymous public bucket."""
    keys: list[str] = []
    continuation: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if continuation:
            params["continuation-token"] = continuation
        url = (
            f"https://{bucket}.s3.amazonaws.com/?"
            + urllib.parse.urlencode(params)
        )
        with urllib.request.urlopen(url, timeout=30.0) as resp:
            tree = ET.parse(resp)
        root = tree.getroot()
        for c in root.findall(f"{S3_LIST_NS}Contents"):
            key = c.findtext(f"{S3_LIST_NS}Key")
            if key:
                keys.append(key)
        is_truncated = root.findtext(f"{S3_LIST_NS}IsTruncated") == "true"
        if not is_truncated:
            break
        continuation = root.findtext(f"{S3_LIST_NS}NextContinuationToken")
    return keys


# ---------------------------------------------------------------------------
# Bbox / period helpers
# ---------------------------------------------------------------------------

def _bboxes_intersect(a: Bbox, b: Bbox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


# DMSP child-catalog names look like F162005_catalog.json
_DMSP_PERIOD_RE = re.compile(r"F\d{2,3}(?P<year>\d{4})")
# VIIRS child-catalog names look like 201501_catalog.json
_VIIRS_PERIOD_RE = re.compile(r"^(?P<yyyymm>\d{6})$")

# Date tokens embedded directly in item filenames (used to prefilter before
# fetching item JSONs — saves thousands of HTTP requests on tight date windows)
# DMSP filename: F{sat}{YYYY}{MM}{DD}{HHMM}.night.OIS.vis.co.json
_DMSP_DATE_RE = re.compile(r"F\d{2,3}(?P<yyyymmdd>\d{8})\d{4}\.night\.OIS")
# VIIRS filename: SVDNB_npp_d{YYYYMMDD}_t{...}_..._noaa_ops.rade9.co.json
_VIIRS_DATE_RE = re.compile(r"_d(?P<yyyymmdd>\d{8})_t")


def _period_in_range(period_id: str, sensor: str, start: datetime, end: datetime) -> bool:
    """Cheap prefilter on child-catalog names before we open them."""
    if sensor == SENSOR_DMSP:
        m = _DMSP_PERIOD_RE.match(period_id)
        if not m:
            return False
        year = int(m.group("year"))
        return year >= start.year and year <= end.year
    if sensor == SENSOR_VIIRS:
        m = _VIIRS_PERIOD_RE.match(period_id)
        if not m:
            return False
        yyyymm = m.group("yyyymm")
        period_dt = datetime(int(yyyymm[:4]), int(yyyymm[4:6]), 1, tzinfo=timezone.utc)
        # keep months whose first day is within [start_month_first, end_month_first]
        start_month = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
        end_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
        return start_month <= period_dt <= end_month
    raise ValueError(f"Unknown sensor: {sensor}")


def _parse_item_datetime(item: dict) -> datetime:
    s = item["properties"]["datetime"]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _date_from_item_url(url: str, sensor: str) -> Optional[datetime]:
    """Extract the calendar date from an item URL filename, or None.

    Used as a cheap prefilter before fetching item JSONs.
    """
    name = url.rsplit("/", 1)[-1]
    if sensor == SENSOR_DMSP:
        m = _DMSP_DATE_RE.search(name)
    elif sensor == SENSOR_VIIRS:
        m = _VIIRS_DATE_RE.search(name)
    else:
        return None
    if not m:
        return None
    s = m.group("yyyymmdd")
    try:
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# STAC catalog walker
# ---------------------------------------------------------------------------

class STACCatalogClient:
    """Walks the static WB-LEN STAC catalog tree.

    The catalogs are plain JSON files in S3 (no STAC API), so we use stdlib
    HTTP and do our own filtering. Output is a stream of STAC item dicts that
    intersect the ROI in space and lie inside the date range.

    For DMSP, an optional `dmsp_preferred_sats` mapping (same shape as
    `harmonizer.config.DMSP_PREFERRED_SATS`) restricts the walk to only the
    `F{sat}{year}` child catalogs that match the preferred-satellite-per-year
    convention from Li et al. 2017 — keeps downstream monthly composites
    single-satellite so DMSPstepwise can apply consistent coefficients.
    """

    def __init__(
        self,
        sensor: str,
        max_workers: int = 16,
        dmsp_preferred_sats: Optional[dict[str, list[str]]] = None,
    ):
        if sensor not in SENSOR_CONFIGS:
            raise ValueError(f"Unknown sensor: {sensor}")
        self.sensor = sensor
        self.cfg: SensorConfig = SENSOR_CONFIGS[sensor]
        self.max_workers = max_workers
        self._dmsp_allowed_sat_years: Optional[set[str]] = None
        if dmsp_preferred_sats is not None and sensor == SENSOR_DMSP:
            self._dmsp_allowed_sat_years = {
                f"{sat}{year}"
                for sat, years in dmsp_preferred_sats.items()
                for year in years
            }

    def _resolve(self, base_url: str, href: str) -> str:
        # urljoin handles "./foo" against ".../catalog.json" correctly
        # (treating the JSON as a file, not a directory).
        return urllib.parse.urljoin(base_url, href)

    def _period_catalog_urls(self, start: datetime, end: datetime) -> list[str]:
        cat = _fetch_json(self.cfg.catalog_url)
        urls: list[str] = []
        for link in cat.get("links", []):
            if link.get("rel") != "child":
                continue
            href = link["href"]
            child_url = self._resolve(self.cfg.catalog_url, href)
            # period_id is the last directory segment in the href, e.g. F162005 or 201501
            period_id = Path(urllib.parse.urlparse(child_url).path).parent.name
            if not _period_in_range(period_id, self.sensor, start, end):
                continue
            if (
                self._dmsp_allowed_sat_years is not None
                and period_id not in self._dmsp_allowed_sat_years
            ):
                continue
            urls.append(child_url)
        return urls

    def _item_urls_in_period(self, period_url: str) -> list[str]:
        cat = _fetch_json(period_url)
        return [
            self._resolve(period_url, link["href"])
            for link in cat.get("links", [])
            if link.get("rel") == "item"
        ]

    def find_items(
        self, roi_bbox: Bbox, start: datetime, end: datetime
    ) -> Iterator[dict]:
        """Yield STAC item dicts intersecting roi_bbox in [start, end]."""
        period_urls = self._period_catalog_urls(start, end)
        log.info("%s: %d period catalogs in range", self.sensor, len(period_urls))

        # Fetch period catalogs in parallel to discover item URLs.
        item_urls: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for urls in pool.map(self._item_urls_in_period, period_urls):
                item_urls.extend(urls)
        log.info("%s: %d items before filename date prefilter", self.sensor, len(item_urls))

        # Cheap prefilter on date tokens embedded in the filename — avoids
        # fetching thousands of item JSONs we'd discard on tight date windows.
        start_d = start.date()
        end_d = end.date()
        prefiltered: list[str] = []
        for url in tqdm(item_urls, desc=f"{self.sensor} URL prefilter", unit="url", leave=False):
            d = _date_from_item_url(url, self.sensor)
            if d is None:
                prefiltered.append(url)
            elif start_d <= d.date() <= end_d:
                prefiltered.append(url)
        item_urls = prefiltered
        log.info("%s: %d items after filename date prefilter", self.sensor, len(item_urls))

        # Fetch each item JSON, filter by bbox + datetime.
        def _maybe_item(url: str) -> Optional[dict]:
            try:
                item = _fetch_json(url)
            except Exception as e:  # pragma: no cover
                log.warning("failed fetching %s: %s", url, e)
                return None
            if "bbox" not in item:
                return None
            if not _bboxes_intersect(tuple(item["bbox"]), roi_bbox):
                return None
            dt = _parse_item_datetime(item)
            if not (start <= dt <= end):
                return None
            return item

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_maybe_item, u): u for u in item_urls}
            with tqdm(total=len(item_urls), desc=f"{self.sensor} fetch items", unit="item") as pbar:
                for fut in as_completed(futures):
                    item = fut.result()
                    if item is not None:
                        yield item
                    pbar.update(1)


# ---------------------------------------------------------------------------
# OrbitRef + triplet resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitRef:
    """A single orbit segment with its three layers resolved to URLs."""

    sensor: str
    orbit_id: str  # canonical, sensor-agnostic key (no extension)
    datetime: datetime
    bbox: Bbox
    radiance_url: str
    li_url: str
    flag_url: str


def _orbit_id_for_dmsp(radiance_url: str) -> str:
    # .../F16200501010040.night.OIS.vis.co.tif -> F16200501010040
    name = Path(urllib.parse.urlparse(radiance_url).path).name
    return name.split(".", 1)[0]


def _orbit_id_for_viirs(radiance_url: str) -> str:
    return viirs_orbit_key(radiance_url)


def _viirs_li_url(radiance_url: str) -> str:
    """Find the matching GDNBO LI file by prefix-listing the bucket.

    The SVDNB radiance and GDNBO LI files share an orbit identifier
    (`npp_d…_t…_e…_b…`) but have different processing-timestamp suffixes, so
    plain string substitution doesn't work. We list the bucket for the orbit
    prefix and pick the one matching `*.li.co.tif`.
    """
    # Extract S3 prefix from the URL. URL is .../{YYYYMM}/SVDNB_..._noaa_ops.rade9.co.tif
    parsed = urllib.parse.urlparse(radiance_url)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected radiance URL: {radiance_url}")
    period, _filename = parts
    orbit = viirs_orbit_key(radiance_url)
    keys = _list_s3_keys(f"{period}/GDNBO_{orbit}_")
    matches = [k for k in keys if k.endswith(".li.co.tif")]
    if not matches:
        raise FileNotFoundError(
            f"No GDNBO li.co.tif found for orbit {orbit} in {period}/"
        )
    if len(matches) > 1:
        log.warning("multiple LI candidates for orbit %s; using first", orbit)
    return f"{S3_HTTPS_BASE}/{matches[0]}"


def orbitref_from_item(item: dict, sensor: str) -> OrbitRef:
    cfg = SENSOR_CONFIGS[sensor]
    radiance_url = item["assets"]["image"]["href"]
    flag_url = cfg.flag_from_radiance(radiance_url)
    if sensor == SENSOR_DMSP:
        li_url = cfg.li_from_radiance(radiance_url)
        orbit_id = _orbit_id_for_dmsp(radiance_url)
    elif sensor == SENSOR_VIIRS:
        li_url = _viirs_li_url(radiance_url)
        orbit_id = _orbit_id_for_viirs(radiance_url)
    else:  # pragma: no cover
        raise ValueError(sensor)
    return OrbitRef(
        sensor=sensor,
        orbit_id=orbit_id,
        datetime=_parse_item_datetime(item),
        bbox=tuple(item["bbox"]),
        radiance_url=radiance_url,
        li_url=li_url,
        flag_url=flag_url,
    )


# ---------------------------------------------------------------------------
# Windowed COG reader
# ---------------------------------------------------------------------------

# GDAL/rasterio environment for efficient anonymous COG access.
# The *_TIMEOUT and LOW_SPEED_* settings are load-bearing: without them a single
# stalled S3 connection will block its worker thread indefinitely, which (under
# concurrency) can deadlock the whole pool.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_LOW_SPEED_TIME": "30",
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1000",
}

_READ_RETRIES = 3
_READ_BACKOFF_SECONDS = 2.0


class WindowedCOGReader:
    """Reads a small ROI window out of a remote COG and writes it locally.

    Cached outputs are keyed by sensor + roi-slug + period + orbit, so cache
    paths from different ROIs never collide. Without the slug, a windowed
    output written for ROI A would be silently re-used by ROI B and produce
    a raster that only contains data over their intersection — a hard-to-spot
    science bug downstream of compositing.
    """

    def __init__(self, cache_dir: Path, roi_bbox: Bbox):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.roi_bbox = tuple(roi_bbox)
        from harmonizer.utils import roi_slug
        self._roi_slug = roi_slug(self.roi_bbox)

    def cache_path(self, orbit: OrbitRef, layer: str) -> Path:
        """Layer is one of {'radiance', 'li', 'flag'}."""
        period = orbit.datetime.strftime("%Y%m")
        sub = self.cache_dir / orbit.sensor / self._roi_slug / period
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{orbit.orbit_id}.{layer}.tif"

    def read_window(self, url: str, roi_bbox: Bbox, dst_path: Path) -> Optional[Path]:
        """Read just the ROI window of a remote COG and write to dst_path.

        Returns dst_path on success, None if the ROI does not overlap the COG.
        Bounded retry on transient failures; the GDAL timeouts in _GDAL_ENV
        ensure each attempt fails fast rather than hanging the worker.
        """
        # Imported lazily so the module can be imported in environments without
        # rasterio (the catalog walker still works).
        import time

        import rasterio
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import transform_bounds

        if dst_path.exists() and dst_path.stat().st_size > 0:
            return dst_path

        last_exc: Optional[Exception] = None
        for attempt in range(_READ_RETRIES):
            try:
                with rasterio.Env(**_GDAL_ENV):
                    with rasterio.open(url) as src:
                        src_bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                        if not _bboxes_intersect(src_bounds_4326, roi_bbox):
                            return None
                        # Project ROI bbox into source CRS, then to a pixel window.
                        xmin, ymin, xmax, ymax = transform_bounds(
                            "EPSG:4326", src.crs, *roi_bbox
                        )
                        window = from_bounds(
                            xmin, ymin, xmax, ymax, transform=src.transform
                        ).round_offsets().round_lengths()
                        # Clip to source extent.
                        full = Window(0, 0, src.width, src.height)
                        window = window.intersection(full)
                        if window.width <= 0 or window.height <= 0:
                            return None
                        data = src.read(1, window=window)
                        profile = src.profile.copy()
                        profile.update(
                            height=int(window.height),
                            width=int(window.width),
                            transform=src.window_transform(window),
                            driver="GTiff",
                            compress="deflate",
                            tiled=True,
                        )
                        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
                        with rasterio.open(tmp_path, "w", **profile) as dst:
                            dst.write(data, 1)
                        os.replace(tmp_path, dst_path)
                return dst_path
            except Exception as e:
                last_exc = e
                if attempt + 1 < _READ_RETRIES:
                    time.sleep(_READ_BACKOFF_SECONDS * (attempt + 1))
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _process_one_orbit(
    item: dict,
    sensor: str,
    roi_bbox: Bbox,
    reader: "WindowedCOGReader",
    layers: list[str],
) -> Optional[dict]:
    """Resolve the (radiance, li, flag) triplet for one STAC item and read each
    layer's ROI window into the local cache. Returns the record dict, or None
    if we can't even resolve the orbit (e.g. VIIRS LI lookup failed).

    Each layer fetch is wrapped: a single failed layer doesn't drop the orbit;
    the bad layer is just None in the returned record. Compositing later
    handles None gracefully.
    """
    try:
        orbit = orbitref_from_item(item, sensor)
    except Exception as e:
        log.warning("failed resolving orbit triplet: %s", e)
        return None
    record: dict = {"orbit": orbit}
    urls = {
        "radiance": orbit.radiance_url,
        "li": orbit.li_url,
        "flag": orbit.flag_url,
    }
    for layer in layers:
        dst = reader.cache_path(orbit, layer)
        try:
            record[layer] = reader.read_window(urls[layer], roi_bbox, dst)
        except Exception as e:
            log.warning("failed reading %s for %s: %s", layer, orbit.orbit_id, e)
            record[layer] = None
    return record


def ingest(
    roi_bbox: Bbox,
    start: datetime,
    end: datetime,
    sensor: str,
    cache_dir: Path,
    max_orbits: Optional[int] = None,
    skip_layers: Iterable[str] = (),
    dmsp_preferred_sats: Optional[dict[str, list[str]]] = None,
    max_workers: int = 32,
) -> list[dict]:
    """Resolve and locally cache all orbit triplets matching ROI + date range.

    Per-orbit work (resolving the LI prefix lookup for VIIRS, then doing
    windowed COG reads for radiance/li/flag) is dispatched across a thread
    pool — every step is HTTP-bound, so threads parallelize cleanly. Cache
    hits in `WindowedCOGReader.read_window` short-circuit, so re-running over
    a wider date range is incremental.

    Returns a list of dicts:
        {"orbit": OrbitRef, "radiance": Path, "li": Path, "flag": Path}
    where Path entries may be None if the layer didn't overlap the ROI or was
    skipped via skip_layers. Output ordering reflects completion order (results
    are collected via as_completed), so a single slow orbit cannot stall reporting
    or downstream consumption of finished records.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    client = STACCatalogClient(sensor, dmsp_preferred_sats=dmsp_preferred_sats)
    reader = WindowedCOGReader(cache_dir, roi_bbox)

    skip = set(skip_layers)
    layers = [l for l in ("radiance", "li", "flag") if l not in skip]

    items = list(client.find_items(roi_bbox, start, end))
    if max_orbits is not None:
        items = items[:max_orbits]
    log.info("%s: dispatching %d orbits across %d worker(s)", sensor, len(items), max_workers)

    def _worker(item):
        return _process_one_orbit(item, sensor, roi_bbox, reader, layers)

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, item) for item in items]
        with tqdm(total=len(items), desc=f"{sensor} orbits", unit="orbit") as pbar:
            for fut in as_completed(futures):
                try:
                    record = fut.result()
                except Exception as e:
                    log.warning("orbit worker raised: %s", e)
                    record = None
                if record is not None:
                    out.append(record)
                pbar.update(1)
    log.info("%s: ingest complete (%d records)", sensor, len(out))
    return out
