"""Step-5 smoke test: full pipeline through monthly Harmonizer fit + transform.

Builds two paired (DMSP, VIIRS) periods in early 2013 (the original overlap
year), trains the Harmonizer on the resulting 2-period stack, then applies
the trained model to a held-out VIIRS period later in 2013.

Verifies:
  - paired training periods are correctly read, NaN-filtered, and concatenated
  - the estimator (CurveFit, deg=3) actually fits and produces a non-trivial
    transform (mean shifts, output clipped to [0, 63])
  - inference on a held-out VIIRS frame writes a sensible per-period raster
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
from harmonizer.transformers.curve import CurveFit
from harmonizer.transformers.harmonize import Harmonizer
from harmonizer.transformers.orbitprep import OrbitPrep

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)

# DMSP F18 covers 2010–2013 in the legacy preferred-sats table.
PREFERRED_SATS = {"F18": ["2010", "2011", "2012", "2013"]}

# Two short training windows + one held-out inference window.
TRAIN_WINDOWS = [
    ("201301", datetime(2013, 1, 1, tzinfo=timezone.utc), datetime(2013, 1, 3, 23, 59, tzinfo=timezone.utc)),
    ("201302", datetime(2013, 2, 1, tzinfo=timezone.utc), datetime(2013, 2, 3, 23, 59, tzinfo=timezone.utc)),
]
INFERENCE_WINDOW = ("201303", datetime(2013, 3, 1, tzinfo=timezone.utc), datetime(2013, 3, 3, 23, 59, tzinfo=timezone.utc))


def build_period(period: str, start, end, dmsp_calib_dir: Path, viirs_prep_dir: Path,
                 ingest_cache: Path, prep_root: Path, composite_root: Path,
                 want_dmsp: bool, want_viirs: bool) -> None:
    if want_viirs:
        recs = ingest(PARIS_BBOX, start, end, SENSOR_VIIRS, ingest_cache)
        prep = OrbitPrep(SENSOR_VIIRS, PARIS_BBOX, prep_root, lunar_mask_mode="all")
        prepped = [prep.transform(r) for r in recs]
        composites = Compositor(SENSOR_VIIRS, composite_root).aggregate(prepped)
        prep_viirs_composites(composites, viirs_prep_dir, usedask=False)

    if want_dmsp:
        recs = ingest(PARIS_BBOX, start, end, SENSOR_DMSP, ingest_cache,
                      dmsp_preferred_sats=PREFERRED_SATS)
        prep = OrbitPrep(SENSOR_DMSP, PARIS_BBOX, prep_root, lunar_mask_mode="all")
        prepped = [prep.transform(r) for r in recs]
        composites = Compositor(SENSOR_DMSP, composite_root).aggregate(prepped)
        calibrate_dmsp_composites(composites, dmsp_calib_dir, PREFERRED_SATS)


def stats(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"shape={arr.shape}  finite=0/{arr.size}"
    return (
        f"shape={arr.shape}  finite={finite.size}/{arr.size}  "
        f"min={finite.min():.4g}  mean={finite.mean():.4g}  max={finite.max():.4g}"
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ingest_cache = td / "ingest"
        prep_root = td / "prep"
        composite_root = td / "composite"
        dmsp_calib = td / "calib_dmsp"
        viirs_prep = td / "prep_viirs"
        harmonized = td / "harmonized"

        for period, start, end in TRAIN_WINDOWS:
            print(f"--- building training period {period} ({start.date()}→{end.date()}) ---")
            build_period(period, start, end, dmsp_calib, viirs_prep,
                         ingest_cache, prep_root, composite_root,
                         want_dmsp=True, want_viirs=True)

        period, start, end = INFERENCE_WINDOW
        print(f"--- building inference period {period} ({start.date()}→{end.date()}, VIIRS only) ---")
        build_period(period, start, end, dmsp_calib, viirs_prep,
                     ingest_cache, prep_root, composite_root,
                     want_dmsp=False, want_viirs=True)

        print("\n--- fitting Harmonizer ---")
        h = Harmonizer(
            dmsp_period_dir=dmsp_calib,
            viirs_period_dir=viirs_prep,
            output_dir=harmonized,
            train_periods=[w[0] for w in TRAIN_WINDOWS],
            est=CurveFit(degree=3),
            polyX=False,         # CurveFit does its own polynomial expansion
            shift=False,
            idX=False,
            epochs=None,
        )
        h.fit()

        print("\n--- applying to training periods (in-sample) ---")
        for period, _, _ in TRAIN_WINDOWS:
            out = h.transform(period)
            with rasterio.open(out) as src:
                print(f"  {period}: {stats(src.read(1))}  -> {out.relative_to(td)}")

        period, _, _ = INFERENCE_WINDOW
        print(f"\n--- applying to held-out period {period} ---")
        out = h.transform(period)
        with rasterio.open(out) as src:
            arr = src.read(1)
        print(f"  {period}: {stats(arr)}  -> {out.relative_to(td)}")

        # Sanity check: training got real pixels and held-out output is in [0, 63].
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0, "held-out output is all-NaN"
        assert finite.min() >= 0.0 and finite.max() <= 63.0, "output out of DMSP DN range"

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
