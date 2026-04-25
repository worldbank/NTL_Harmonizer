"""Step-6 end-to-end smoke: invoke harmonizer.main with a tight ROI/window.

Trains on Feb 2013, infers on Mar 2013 (and the train window). Uses CurveFit
to avoid the xgboost dependency. Uses lunar-mode "all" so we don't depend on
moon phase to produce data in a 3-day window.

Verifies:
  - main() runs end-to-end without error
  - per-period harmonized rasters land at OUTPUT/{trial}/{period}/radiance.tif
  - diagnostic plots and metrics file are written under RESULTS/{trial}/
"""
from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from harmonizer.config import OUTPUT, RESULTS, ARTIFACTS, INGEST_CACHE, COMPOSITE_DIR, PREP_DIR, CALIB_DIR, VIIRS_PREP_DIR
from harmonizer.main import main as run_main
from harmonizer.transformers.curve import CurveFit


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    trialname = "smoke_step6"
    # Clean previous run artifacts so we exercise the build paths.
    for d in (OUTPUT / trialname, RESULTS / trialname):
        if d.exists():
            shutil.rmtree(d)

    run_main(
        trialname=trialname,
        roi_bbox=(2.0, 48.5, 3.0, 49.5),  # tight bbox over Paris
        start=datetime(2013, 2, 1, tzinfo=timezone.utc),
        end=datetime(2013, 3, 3, 23, 59, tzinfo=timezone.utc),
        train_year=2013,
        lunar_mode="all",
        est=CurveFit(degree=3),
        polyX=False,  # CurveFit does its own polynomial expansion
    )

    # Sanity checks
    trialout = OUTPUT / trialname
    trialresults = RESULTS / trialname
    artifact = ARTIFACTS / f"{trialname}_harmonizer.dill"

    rasters = sorted(trialout.glob("*/radiance.tif"))
    assert rasters, f"no harmonized rasters under {trialout}"
    print(f"\nharmonized rasters: {[str(p.relative_to(trialout)) for p in rasters]}")

    plots = sorted(trialresults.glob("*"))
    assert plots, f"no diagnostic outputs under {trialresults}"
    print(f"diagnostic outputs: {[p.name for p in plots]}")

    assert artifact.exists(), f"missing trained artifact at {artifact}"
    print(f"artifact: {artifact} ({artifact.stat().st_size} bytes)")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
