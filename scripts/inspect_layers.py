"""Inspect the radiance/li/flag clips produced by step 1 to confirm dtypes,
nodata, and the actual bit values present in the flag layer over Paris.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from harmonizer.constants import (
    SENSOR_DMSP,
    SENSOR_VIIRS,
    SENSOR_CONFIGS,
)
from harmonizer.ingest import (
    STACCatalogClient,
    WindowedCOGReader,
    orbitref_from_item,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)


def fetch_one(sensor: str, start, end, cache: Path) -> dict:
    client = STACCatalogClient(sensor)
    items = list(client.find_items(PARIS_BBOX, start, end))
    if not items:
        raise SystemExit(f"no {sensor} items found")
    orbit = orbitref_from_item(items[0], sensor)
    reader = WindowedCOGReader(cache)
    out = {"orbit": orbit}
    for layer, url in (
        ("radiance", orbit.radiance_url),
        ("li", orbit.li_url),
        ("flag", orbit.flag_url),
    ):
        out[layer] = reader.read_window(url, PARIS_BBOX, reader.cache_path(orbit, layer))
    return out


def describe(record: dict, sensor: str) -> None:
    print(f"\n=== {sensor} | orbit {record['orbit'].orbit_id} ===")
    cfg = SENSOR_CONFIGS[sensor]
    for layer in ("radiance", "li", "flag"):
        path = record[layer]
        with rasterio.open(path) as src:
            arr = src.read(1)
            print(
                f"  {layer:9s} {path.name}  "
                f"dtype={arr.dtype}  shape={arr.shape}  nodata={src.nodata}  "
                f"min={arr.min()}  max={arr.max()}  unique<=10={len(np.unique(arr))<=10}"
            )
            if layer == "flag":
                vals, counts = np.unique(arr, return_counts=True)
                print(f"    distinct flag values (top 8 by count):")
                order = np.argsort(-counts)[:8]
                for i in order:
                    v = int(vals[i])
                    bits = ",".join(str(b) for b in range(16) if (v >> b) & 1) or "(none)"
                    print(f"      0x{v:04x} = {v:5d}  bits set: {bits}    n={counts[i]}")
                # specifically check the zero-lunar bit
                zb = cfg.zero_lunar_bit
                zero_lunar_set = ((arr.astype(np.int64) >> zb) & 1).astype(bool)
                pct = 100.0 * zero_lunar_set.sum() / arr.size
                print(f"    bit {zb} (zero-lunar) set in {pct:.1f}% of pixels")
            if layer == "li":
                with_data = arr[arr != cfg.li_nodata]
                if with_data.size:
                    print(
                        f"    LI (non-nodata): min={with_data.min():.3g}  "
                        f"mean={with_data.mean():.3g}  max={with_data.max():.3g}  "
                        f"({with_data.size}/{arr.size} pixels)"
                    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        viirs = fetch_one(
            SENSOR_VIIRS,
            datetime(2015, 1, 1, tzinfo=timezone.utc),
            datetime(2015, 1, 1, 23, 59, tzinfo=timezone.utc),
            cache,
        )
        describe(viirs, SENSOR_VIIRS)
        dmsp = fetch_one(
            SENSOR_DMSP,
            datetime(2005, 1, 1, tzinfo=timezone.utc),
            datetime(2005, 1, 1, 23, 59, tzinfo=timezone.utc),
            cache,
        )
        describe(dmsp, SENSOR_DMSP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
