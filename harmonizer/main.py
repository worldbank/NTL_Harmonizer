"""WB-LEN harmonizer entry point.

Runs the full pipeline:

    1. STAC-windowed ingest of nightly orbit triplets (radiance / LI / flag)
    2. Per-orbit cleaning + warp to a common ROI grid
    3. Per-period (default monthly) compositing
    4. DMSPstepwise intercalibration + VIIRSprep preprocessing
    5. Harmonizer fit on the training year's monthly stack
    6. Inference across every available VIIRS period
    7. Diagnostics

Inputs come from `s3://globalnightlight/` — there is no manual download step.
Caches under `data/cache/...` make reruns incremental.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from harmonizer.calibrate import calibrate_dmsp_composites, prep_viirs_composites
from harmonizer.composite import Compositor
from harmonizer.config import (
    ARTIFACTS,
    CALIB_DIR,
    COMPOSITE_DIR,
    DMSP_PREFERRED_SATS,
    DOWNSAMPLEVIIRS,
    END_DATE,
    INGEST_CACHE,
    LUNAR_MASK_MODE,
    OUTPUT,
    PERIOD_FORMAT,
    PREP_DIR,
    RESULTS,
    ROIPATH,
    SAMPLEMETHOD,
    START_DATE,
    TRAIN_YEAR,
    VIIRS_PREP_DIR,
)
from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS
from harmonizer.diagnostics import run_diagnostics
from harmonizer.ingest import ingest
from harmonizer.transformers.gbm import XGB
from harmonizer.transformers.harmonize import Harmonizer, save_obj
from harmonizer.transformers.orbitprep import OrbitPrep
from harmonizer.utils import roi_bbox_from_path

log = logging.getLogger(__name__)


def _build_sensor_composites(
    sensor: str,
    roi_bbox,
    start: datetime,
    end: datetime,
    lunar_mode: str,
    period_format: str,
) -> tuple[list[dict], str]:
    """Run ingest → orbitprep → composite for one sensor.

    Returns ``(composites, roi_slug)``. The slug identifies the cache
    namespace used for this sensor's outputs and is reused downstream by
    calibrate / viirsprep so the whole pipeline shares a consistent
    ROI-keyed path layout.
    """
    extra = {"dmsp_preferred_sats": DMSP_PREFERRED_SATS} if sensor == SENSOR_DMSP else {}
    records = ingest(roi_bbox, start, end, sensor, INGEST_CACHE, **extra)
    log.info("%s: ingested %d orbits", sensor, len(records))

    prep = OrbitPrep(sensor, roi_bbox, PREP_DIR, lunar_mask_mode=lunar_mode)
    prepped = [prep.transform(r) for r in records]

    composites = Compositor(
        sensor, COMPOSITE_DIR, roi_slug=prep.roi_slug, period_format=period_format,
    ).aggregate(prepped)
    log.info("%s: built %d composite period(s)", sensor, len(composites))
    return composites, prep.roi_slug


def main(
    trialname: str,
    roi_bbox,
    start: datetime,
    end: datetime,
    train_year: int,
    lunar_mode: str = LUNAR_MASK_MODE,
    period_format: str = PERIOD_FORMAT,
    est=None,
    polyX: bool = True,
    shift: bool = False,
    idX: bool = False,
    skip_diagnostics: bool = False,
) -> None:
    if est is None:
        est = XGB()

    trialout = OUTPUT / trialname
    trialresults = RESULTS / trialname
    trialout.mkdir(parents=True, exist_ok=True)
    trialresults.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== {trialname}: {start.date()} → {end.date()}, train={train_year}, mode={lunar_mode!r} ===")

    # 1–3. Ingest, orbitprep, composite for both sensors.
    composites_by_sensor: dict[str, list[dict]] = {}
    slug_by_sensor: dict[str, str] = {}
    for sensor in (SENSOR_DMSP, SENSOR_VIIRS):
        t = time.time()
        composites_by_sensor[sensor], slug_by_sensor[sensor] = _build_sensor_composites(
            sensor, roi_bbox, start, end, lunar_mode, period_format,
        )
        print(f"  {sensor}: ingest+prep+composite in {time.time() - t:.1f}s")

    # 4. DMSP intercalibration + VIIRS preprocessing.
    t = time.time()
    dmsp_calibrated = calibrate_dmsp_composites(
        composites_by_sensor[SENSOR_DMSP], CALIB_DIR, DMSP_PREFERRED_SATS,
        roi_slug=slug_by_sensor[SENSOR_DMSP],
    )
    viirs_prepped = prep_viirs_composites(
        composites_by_sensor[SENSOR_VIIRS], VIIRS_PREP_DIR,
        roi_slug=slug_by_sensor[SENSOR_VIIRS],
    )
    print(f"  calibrate + prep in {time.time() - t:.1f}s")

    # 5. Determine training periods (overlap year, intersected with what we have).
    dmsp_periods = {c["period"] for c in dmsp_calibrated}
    viirs_periods = {c["period"] for c in viirs_prepped}
    train_periods = sorted(
        p for p in (dmsp_periods & viirs_periods) if p.startswith(str(train_year))
    )
    if not train_periods:
        raise RuntimeError(
            f"no overlapping periods in train year {train_year}: "
            f"dmsp periods={sorted(dmsp_periods)} viirs periods={sorted(viirs_periods)}"
        )
    print(f"  training on {len(train_periods)} period(s): {train_periods}")

    # 6. Fit + apply.
    t = time.time()
    harmonizer = Harmonizer(
        dmsp_period_dir=CALIB_DIR / slug_by_sensor[SENSOR_DMSP],
        viirs_period_dir=VIIRS_PREP_DIR / slug_by_sensor[SENSOR_VIIRS],
        output_dir=trialout,
        train_periods=train_periods,
        downsampleVIIRS=DOWNSAMPLEVIIRS,
        samplemethod=SAMPLEMETHOD,
        polyX=polyX,
        shift=shift,
        idX=idX,
        epochs=100,
        est=est,
        opath=ARTIFACTS / f"{trialname}_harmonizer.dill",
    )
    harmonizer.fit()
    save_obj(harmonizer, harmonizer.opath)
    print(f"  fit in {time.time() - t:.1f}s")

    t = time.time()
    inference_periods = sorted(viirs_periods)
    n_emitted = 0
    for period in inference_periods:
        out = harmonizer.transform(period)
        if out is not None:
            n_emitted += 1
    print(f"  inference: {n_emitted}/{len(inference_periods)} periods emitted in {time.time() - t:.1f}s")

    # 7. Diagnostics.
    if not skip_diagnostics:
        t = time.time()
        run_diagnostics(
            trialout=trialout,
            trialresults=trialresults,
            dmsp_calib_dir=CALIB_DIR / slug_by_sensor[SENSOR_DMSP],
            viirs_prep_dir=VIIRS_PREP_DIR / slug_by_sensor[SENSOR_VIIRS],
            train_periods=train_periods,
            inference_periods=inference_periods,
        )
        print(f"  diagnostics in {time.time() - t:.1f}s")

    print(f"DONE in {time.time() - t0:.1f}s — outputs at {trialout}")


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--name", default="test", help="trial name")
    p.add_argument(
        "--roi", default=str(ROIPATH),
        help="ROI shapefile path or 'xmin,ymin,xmax,ymax' bbox in EPSG:4326",
    )
    p.add_argument("--start", default=START_DATE.isoformat(), type=_parse_date)
    p.add_argument("--end", default=END_DATE.isoformat(), type=_parse_date)
    p.add_argument("--train-year", type=int, default=TRAIN_YEAR)
    p.add_argument(
        "--lunar-mode", default=LUNAR_MASK_MODE,
        choices=["zero", "low", "all"],
    )
    p.add_argument("--period-format", default=PERIOD_FORMAT)
    p.add_argument(
        "--skip-diagnostics", action="store_true",
        help="skip the post-run plots/metrics step",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = get_args()
    bbox = roi_bbox_from_path(args.roi)
    main(
        trialname=args.name,
        roi_bbox=bbox,
        start=args.start,
        end=args.end,
        train_year=args.train_year,
        lunar_mode=args.lunar_mode,
        period_format=args.period_format,
        skip_diagnostics=args.skip_diagnostics,
    )
