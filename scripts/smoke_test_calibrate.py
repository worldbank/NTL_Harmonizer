"""Step-4 smoke test: composite → DMSPstepwise / VIIRSprep on monthly composites.

Verifies that:
  - the DMSP `dmsp_preferred_sats` ingest filter actually drops non-preferred
    satellite-year catalogs (F152005 in this case) and only walks F162005
  - DMSPstepwise applied with explicit satellite_year produces a sensible
    calibrated raster from a monthly composite
  - VIIRSprep applied to a monthly composite produces a sensible prepped
    raster (damper + Gaussian convolve + log1p)
"""
from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from harmonizer.calibrate import calibrate_dmsp_composites, prep_viirs_composites
from harmonizer.composite import Compositor
from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS
from harmonizer.ingest import ingest
from harmonizer.transformers.orbitprep import OrbitPrep

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)

# Mirror harmonizer.config.DMSP_PREFERRED_SATS so the smoke test is independent.
PREFERRED_SATS = {
    "F10": ["1992", "1993", "1994"],
    "F12": ["1995", "1996"],
    "F14": ["1997", "1998", "1999", "2000", "2001", "2002", "2003"],
    "F16": ["2004", "2005", "2006", "2007", "2008", "2009"],
    "F18": ["2010", "2011", "2012", "2013"],
}


def stats(path: Path) -> str:
    with rasterio.open(path) as src:
        arr = src.read(1)
    if arr.dtype.kind in "iu":
        return (
            f"shape={arr.shape}  dtype={arr.dtype}  "
            f"min={arr.min()}  mean={arr.mean():.3f}  max={arr.max()}"
        )
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"shape={arr.shape}  dtype={arr.dtype}  finite=0/{arr.size}"
    return (
        f"shape={arr.shape}  dtype={arr.dtype}  finite={finite.size}/{arr.size}  "
        f"min={finite.min():.4g}  mean={finite.mean():.4g}  max={finite.max():.4g}"
    )


def run_dmsp(out_root: Path, ingest_cache: Path) -> None:
    print(f"\n{'='*72}\nDMSP step-4 (Jan 1–3 2005, F16 preferred)\n{'='*72}")
    records = ingest(
        PARIS_BBOX,
        datetime(2005, 1, 1, tzinfo=timezone.utc),
        datetime(2005, 1, 3, 23, 59, tzinfo=timezone.utc),
        SENSOR_DMSP,
        ingest_cache,
        dmsp_preferred_sats=PREFERRED_SATS,
    )
    sats_seen = sorted({r["orbit"].orbit_id[:3] for r in records})
    print(f"  ingest: {len(records)} orbits  satellites_seen={sats_seen}")
    assert sats_seen == ["F16"], f"DMSP filter failed: saw {sats_seen}"

    prep = OrbitPrep(SENSOR_DMSP, PARIS_BBOX, out_root / "prep", lunar_mask_mode="zero")
    prepped = [prep.transform(r) for r in records]
    composited = Compositor(SENSOR_DMSP, out_root / "composite").aggregate(prepped)
    print(f"  composite: {len(composited)} period(s)")
    for c in composited:
        print(f"  composite {c['period']}: radiance={stats(c['radiance'])}")

    calibrated = calibrate_dmsp_composites(
        composited, out_root / "calibrated", PREFERRED_SATS
    )
    for c in calibrated:
        print(
            f"  calibrated {c['period']} (sat_year={c['satellite_year']}): "
            f"{stats(c['radiance'])}"
        )


def run_viirs(out_root: Path, ingest_cache: Path) -> None:
    print(f"\n{'='*72}\nVIIRS step-4 (Jan 19–21 2015)\n{'='*72}")
    records = ingest(
        PARIS_BBOX,
        datetime(2015, 1, 19, tzinfo=timezone.utc),
        datetime(2015, 1, 21, 23, 59, tzinfo=timezone.utc),
        SENSOR_VIIRS,
        ingest_cache,
    )
    print(f"  ingest: {len(records)} orbits")

    prep = OrbitPrep(SENSOR_VIIRS, PARIS_BBOX, out_root / "prep", lunar_mask_mode="zero")
    prepped = [prep.transform(r) for r in records]
    composited = Compositor(SENSOR_VIIRS, out_root / "composite").aggregate(prepped)
    print(f"  composite: {len(composited)} period(s)")
    for c in composited:
        print(f"  composite {c['period']}: radiance={stats(c['radiance'])}")

    prepped_v = prep_viirs_composites(
        composited, out_root / "viirs_prepped", usedask=False,
    )
    for c in prepped_v:
        print(f"  prepped {c['period']}: {stats(c['radiance'])}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        ingest_cache = Path(td) / "ingest"
        out_root = Path(td) / "out"
        run_dmsp(out_root, ingest_cache)
        run_viirs(out_root, ingest_cache)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
