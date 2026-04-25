"""Per-period calibration / preprocessing for monthly composites.

Step 4 of the WB-LEN ingestion pipeline. Takes the per-period composite
records produced by `harmonizer.composite.Compositor.aggregate(...)` and
applies the existing transformers (`DMSPstepwise`, `VIIRSprep`) to each
period in turn.

The transformer math is unchanged from the legacy annual-composite pipeline.
What's new is that:

- DMSPstepwise is told the satellite-year explicitly, so it doesn't need to
  recover that identity from a string match against the input path.
- The output is one calibrated/prepped raster per period, keyed by period
  string, rather than one per annual file.

DMSP coverage gap: the published Li 2017 coefficients only cover F14/F15/F16/
F18 for specific years. Periods whose preferred satellite-year falls outside
that set are emitted clip-only (the legacy fallback). That's a known limit
of the existing transformer, not a new one introduced here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from harmonizer.constants import SENSOR_DMSP, SENSOR_VIIRS
from harmonizer.transformers.dmspcalibrate import DMSPstepwise
from harmonizer.transformers.viirsprep import VIIRSprep

log = logging.getLogger(__name__)


def _preferred_sat_for_year(year: str, preferred_sats: dict[str, list[str]]) -> str | None:
    """Return the satellite name (e.g. 'F16') preferred for a given calendar year."""
    for sat, years in preferred_sats.items():
        if year in years:
            return sat
    return None


def calibrate_dmsp_composites(
    composites: Iterable[dict],
    dst_dir: Path,
    preferred_sats: dict[str, list[str]],
) -> list[dict]:
    """Run DMSPstepwise on each per-period composite, picking coefs by sat-year.

    `composites` is the iterable returned by `Compositor.aggregate(...)` for
    the DMSP sensor. Each record has keys: sensor, period, n_orbits, radiance,
    li, obs_count.

    Output records mirror the input but with `radiance` rebound to the
    calibrated raster path under `dst_dir`.
    """
    dst_dir = Path(dst_dir)
    calibrator = DMSPstepwise(dstdir=dst_dir)  # dstdir kept for compatibility
    out: list[dict] = []
    for rec in composites:
        if rec["sensor"] != SENSOR_DMSP:
            raise ValueError(f"expected dmsp record, got {rec['sensor']!r}")
        period = rec["period"]
        year = period[:4]
        sat = _preferred_sat_for_year(year, preferred_sats)
        if sat is None:
            log.warning(
                "no preferred satellite for year %s; skipping DMSP composite %s",
                year, period,
            )
            continue
        sat_year = f"{sat}{year}"
        period_dir = dst_dir / period
        period_dir.mkdir(parents=True, exist_ok=True)
        dst_path = period_dir / "radiance.tif"
        calibrator.transform(rec["radiance"], satellite_year=sat_year, dstpath=dst_path)
        log.info("DMSP %s: applied coefs for %s -> %s", period, sat_year, dst_path)
        out.append({**rec, "radiance": dst_path, "satellite_year": sat_year})
    return out


def prep_viirs_composites(
    composites: Iterable[dict],
    dst_dir: Path,
    pixelradius: int = 5,
    sigma: float = 2.0,
    damperthresh: float = 1.0,
    usedask: bool = False,
    chunks: str | None = "auto",
) -> list[dict]:
    """Run VIIRSprep on each per-period composite radiance raster.

    Default damper / convolution / log-transform parameters match the legacy
    annual-composite pipeline. They may benefit from re-tuning at monthly
    cadence (the input is noisier than an annual composite) — flagged for a
    later sweep.
    """
    dst_dir = Path(dst_dir)
    prepper = VIIRSprep(
        pixelradius=pixelradius,
        sigma=sigma,
        damperthresh=damperthresh,
        usedask=usedask,
        chunks=chunks,
        dstdir=dst_dir,
    )
    out: list[dict] = []
    for rec in composites:
        if rec["sensor"] != SENSOR_VIIRS:
            raise ValueError(f"expected viirs_npp record, got {rec['sensor']!r}")
        period = rec["period"]
        period_dir = dst_dir / period
        period_dir.mkdir(parents=True, exist_ok=True)
        dst_path = period_dir / "radiance.tif"
        prepper.transform(rec["radiance"], dstpath=dst_path)
        log.info("VIIRS %s: prep -> %s", period, dst_path)
        out.append({**rec, "radiance": dst_path})
    return out
