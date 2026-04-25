"""Smoke test for harmonizer.transformers.orbitprep.

Pulls a small handful of orbits over Paris (one VIIRS day, one DMSP day),
runs OrbitPrep in each lunar-mask mode, and reports per-orbit kept-pixel
percentages plus simple stats on the output rasters.

The VIIRS test day (2015-01-01) is moonlit, so:
  - "zero" mode should produce all-NaN VIIRS output (0% kept)
  - "low"  mode should also reject (LI ~0.019 lux > default 0.1 thresh? actually
                                     well below 0.1, so KEEP)
  - "all"  mode should produce non-NaN output

The DMSP test day (2005-01-01) is zero-lunar, so:
  - All modes should produce non-empty output for valid pixels.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS
from harmonizer.ingest import ingest
from harmonizer.transformers.orbitprep import OrbitPrep, make_target_grid

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)


def summarize(path: Path) -> str:
    if path is None or not path.exists():
        return "<missing>"
    with rasterio.open(path) as src:
        arr = src.read(1)
    finite = arr[np.isfinite(arr)]
    n_total = arr.size
    n_finite = finite.size
    if n_finite == 0:
        return f"shape={arr.shape}  finite=0/{n_total}  (all-NaN)"
    return (
        f"shape={arr.shape}  finite={n_finite}/{n_total}  "
        f"min={finite.min():.4g}  mean={finite.mean():.4g}  max={finite.max():.4g}"
    )


def run_sensor(sensor: str, start, end, out_root: Path, ingest_cache: Path) -> None:
    print(f"\n{'='*72}\n{sensor}: {start.date()} → {end.date()}\n{'='*72}")
    records = ingest(PARIS_BBOX, start, end, sensor, ingest_cache)
    print(f"  ingested orbits: {len(records)}")

    for mode in ("zero", "low", "all"):
        prep = OrbitPrep(
            sensor=sensor,
            roi_bbox=PARIS_BBOX,
            dst_dir=out_root / f"mode_{mode}",
            lunar_mask_mode=mode,
        )
        print(f"\n  --- mode={mode!r} ---")
        print(
            f"  target grid: {prep._grid.width}x{prep._grid.height} "
            f"@ {prep._grid.pixel_size_deg*3600:.1f} arc-sec  bounds={prep._grid.bounds}"
        )
        for rec in records:
            out = prep.transform(rec)
            orbit_id = rec["orbit"].orbit_id
            print(f"  {orbit_id}")
            print(f"    radiance: {summarize(out['radiance'])}")
            print(f"    li:       {summarize(out['li'])}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ingest_cache = Path(td) / "ingest"
        prep_out = Path(td) / "orbitprep"
        run_sensor(
            SENSOR_VIIRS,
            datetime(2015, 1, 1, tzinfo=timezone.utc),
            datetime(2015, 1, 1, 23, 59, tzinfo=timezone.utc),
            prep_out, ingest_cache,
        )
        run_sensor(
            SENSOR_DMSP,
            datetime(2005, 1, 1, tzinfo=timezone.utc),
            datetime(2005, 1, 1, 23, 59, tzinfo=timezone.utc),
            prep_out, ingest_cache,
        )
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
