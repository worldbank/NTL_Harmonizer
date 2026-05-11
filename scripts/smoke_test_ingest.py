"""Smoke test for harmonizer.ingest — Step 1.

Stage 1 (no rasterio needed): walks the WB-LEN STAC catalog and resolves
the (radiance, li, flag) triplet for a small ROI/date window. Verifies:
  - period catalog filtering
  - bbox + date filtering of items
  - DMSP layer triplet (string substitution)
  - VIIRS layer triplet (LI prefix lookup)

Stage 2 (rasterio required): does one windowed COG read per layer. Skipped if
rasterio is unavailable.

Run:
    python -m scripts.smoke_test_ingest
"""
from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from harmonizer.ingest import (
    STACCatalogClient,
    WindowedCOGReader,
    orbitref_from_item,
)
from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Tight bbox over Paris: ~2°×2° to keep the smoke test fast.
PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)

# Single-day windows minimize STAC traffic.
VIIRS_START = datetime(2015, 1, 1, 0, 0, tzinfo=timezone.utc)
VIIRS_END = datetime(2015, 1, 1, 23, 59, tzinfo=timezone.utc)
DMSP_START = datetime(2005, 1, 1, 0, 0, tzinfo=timezone.utc)
DMSP_END = datetime(2005, 1, 1, 23, 59, tzinfo=timezone.utc)


def stage1(sensor: str, start, end) -> "OrbitRef":
    print(f"\n=== Stage 1: {sensor} catalog walk ===")
    client = STACCatalogClient(sensor)
    items = list(client.find_items(PARIS_BBOX, start, end))
    print(f"  matched items: {len(items)}")
    if not items:
        raise SystemExit(f"no {sensor} items found over Paris on {start.date()}")
    orbit = orbitref_from_item(items[0], sensor)
    print(f"  orbit_id: {orbit.orbit_id}")
    print(f"  datetime: {orbit.datetime.isoformat()}")
    print(f"  bbox:     {orbit.bbox}")
    print(f"  radiance: {orbit.radiance_url}")
    print(f"  li:       {orbit.li_url}")
    print(f"  flag:     {orbit.flag_url}")
    return orbit


def stage2(orbit: "OrbitRef", cache_dir: Path) -> None:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        print("  (rasterio not installed; skipping windowed read)")
        return
    print(f"\n=== Stage 2: windowed read of {orbit.sensor} {orbit.orbit_id} ===")
    reader = WindowedCOGReader(cache_dir)
    for layer, url in (
        ("radiance", orbit.radiance_url),
        ("li", orbit.li_url),
        ("flag", orbit.flag_url),
    ):
        dst = reader.cache_path(orbit, layer)
        out = reader.read_window(url, PARIS_BBOX, dst)
        if out is None:
            print(f"  {layer}: ROI did not overlap source")
        else:
            size_kb = out.stat().st_size / 1024
            print(f"  {layer}: {out} ({size_kb:.1f} KB)")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        viirs = stage1(SENSOR_VIIRS, VIIRS_START, VIIRS_END)
        stage2(viirs, cache)
        dmsp = stage1(SENSOR_DMSP, DMSP_START, DMSP_END)
        stage2(dmsp, cache)
    print("\nDONE")


if __name__ == "__main__":
    sys.exit(main())
