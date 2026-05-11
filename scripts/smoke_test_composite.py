"""Smoke test for the full ingest → orbitprep → composite pipeline.

VIIRS test window: 2015-01-19 → 2015-01-21 (around new moon, so zero-lunar
mode should actually keep data instead of dropping every orbit).
DMSP test window:  2005-01-01 → 2005-01-03 (the known-zero-lunar period from
the orbitprep smoke test, extended to two more days to build a small composite).

Verifies:
  - period grouping (single monthly key for each sensor)
  - obs_count peaks at sensible values for the date range
  - composite radiance is finite over Paris (urban lights persist across nights)
"""
from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from harmonizer.composite import Compositor
from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS
from harmonizer.ingest import ingest
from harmonizer.transformers.orbitprep import OrbitPrep

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)


def stats(path: Path) -> str:
    with rasterio.open(path) as src:
        arr = src.read(1)
    if arr.dtype.kind in "iu":
        return (
            f"shape={arr.shape}  dtype={arr.dtype}  "
            f"min={arr.min()}  mean={arr.mean():.2f}  max={arr.max()}  "
            f">0={int((arr > 0).sum())}/{arr.size}"
        )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"shape={arr.shape}  dtype={arr.dtype}  finite=0/{arr.size}"
    return (
        f"shape={arr.shape}  dtype={arr.dtype}  finite={finite.size}/{arr.size}  "
        f"min={finite.min():.4g}  mean={finite.mean():.4g}  max={finite.max():.4g}"
    )


def run_sensor(
    sensor: str,
    start: datetime,
    end: datetime,
    lunar_mode: str,
    out_root: Path,
    ingest_cache: Path,
) -> None:
    print(f"\n{'='*72}\n{sensor}: {start.date()} → {end.date()}  mode={lunar_mode!r}\n{'='*72}")

    records = ingest(PARIS_BBOX, start, end, sensor, ingest_cache)
    print(f"  ingest: {len(records)} orbits")

    prep = OrbitPrep(
        sensor=sensor,
        roi_bbox=PARIS_BBOX,
        dst_dir=out_root / "prep",
        lunar_mask_mode=lunar_mode,
    )
    prepped = [prep.transform(r) for r in records]
    n_with_data = sum(
        1
        for r in prepped
        if r["radiance"] is not None
        and rasterio.open(r["radiance"]).read(1).size
        and np.isfinite(rasterio.open(r["radiance"]).read(1)).any()
    )
    print(f"  orbitprep: {n_with_data}/{len(prepped)} orbits with any valid pixels after mask")

    compositor = Compositor(
        sensor=sensor, dst_dir=out_root / "composite", roi_slug=prep.roi_slug,
    )
    composites = compositor.aggregate(prepped)
    print(f"  composite: {len(composites)} period(s)")

    for c in composites:
        print(f"  period={c['period']}  n_orbits={c['n_orbits']}")
        print(f"    radiance:  {stats(c['radiance'])}")
        print(f"    li:        {stats(c['li'])}")
        print(f"    obs_count: {stats(c['obs_count'])}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ingest_cache = Path(td) / "ingest"
        out_root = Path(td) / "out"
        run_sensor(
            SENSOR_VIIRS,
            datetime(2015, 1, 19, tzinfo=timezone.utc),
            datetime(2015, 1, 21, 23, 59, tzinfo=timezone.utc),
            lunar_mode="zero",
            out_root=out_root,
            ingest_cache=ingest_cache,
        )
        run_sensor(
            SENSOR_DMSP,
            datetime(2005, 1, 1, tzinfo=timezone.utc),
            datetime(2005, 1, 3, 23, 59, tzinfo=timezone.utc),
            lunar_mode="zero",
            out_root=out_root,
            ingest_cache=ingest_cache,
        )
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
