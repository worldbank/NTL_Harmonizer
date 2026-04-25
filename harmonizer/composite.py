"""Temporal compositing: reduce per-orbit rasters to per-period composites.

Step 3 of the WB-LEN ingestion pipeline. Given a stream of cleaned per-orbit
records produced by `OrbitPrep`, group them by period (default monthly) and
emit a single composite per period:

    radiance.tif    pixelwise reducer over valid observations (median by default)
    li.tif          pixelwise mean lunar illuminance over valid observations
    obs_count.tif   number of valid observations per pixel (uint16)

All three rasters share the orbit-level common ROI grid, so downstream
harmonization can stack periods without further reprojection.

The reducer ignores NaN; pixels with fewer than `min_obs` valid observations
are emitted as NaN. Median is the EOG composite convention and is robust to
the cloud-edge / transient-light artifacts that nightly imagery is full of;
mean is provided as an option for users who want smoother outputs.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio

from harmonizer.constants import SENSOR_CONFIGS

log = logging.getLogger(__name__)

REDUCERS = {
    "median": np.nanmedian,
    "mean": np.nanmean,
}


@dataclass
class Compositor:
    """Reduce per-orbit rasters to per-period composites.

    Parameters
    ----------
    sensor : "dmsp" | "viirs_npp"
    dst_dir : output root; composites land in {dst_dir}/{sensor}/{period}/
    method : "median" (default) or "mean"
    period_format : strftime pattern for the period key. Default "%Y%m"
                    (monthly). "%Y" for annual, "%Y%m%d" for daily.
    min_obs : minimum number of valid observations required per pixel; pixels
              below this threshold come out as NaN. Default 1.
    """

    sensor: str
    dst_dir: Path
    method: str = "median"
    period_format: str = "%Y%m"
    min_obs: int = 1

    def __post_init__(self):
        if self.sensor not in SENSOR_CONFIGS:
            raise ValueError(f"unknown sensor: {self.sensor}")
        if self.method not in REDUCERS:
            raise ValueError(f"method must be one of {list(REDUCERS)}, got {self.method!r}")
        self.dst_dir = Path(self.dst_dir)

    # ---- API ----------------------------------------------------------

    def aggregate(self, orbit_outputs: Iterable[dict]) -> list[dict]:
        """Composite each period in the input stream. Returns one record per period."""
        groups: dict[str, list[dict]] = defaultdict(list)
        for rec in orbit_outputs:
            if rec.get("radiance") is None or rec.get("li") is None:
                continue
            period = rec["orbit"].datetime.strftime(self.period_format)
            groups[period].append(rec)

        results: list[dict] = []
        for period in sorted(groups):
            results.append(self.composite_period(period, groups[period]))
        return results

    def composite_period(self, period: str, records: list[dict]) -> dict:
        """Reduce one period's worth of orbit records into a composite triplet."""
        out_dir = self.dst_dir / self.sensor / period
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = {
            "radiance": out_dir / "radiance.tif",
            "li": out_dir / "li.tif",
            "obs_count": out_dir / "obs_count.tif",
        }

        # Stack radiance and LI from each orbit. They share the common ROI
        # grid so we just stack arrays.
        radiance_stack: list[np.ndarray] = []
        li_stack: list[np.ndarray] = []
        ref_profile: Optional[dict] = None
        for rec in records:
            with rasterio.open(rec["radiance"]) as src:
                radiance_stack.append(src.read(1))
                if ref_profile is None:
                    ref_profile = src.profile.copy()
            with rasterio.open(rec["li"]) as src:
                li_stack.append(src.read(1))

        radiance = np.stack(radiance_stack, axis=0)
        li = np.stack(li_stack, axis=0)
        # finite mask drives obs_count and gates the reducers
        valid = np.isfinite(radiance)
        obs_count = valid.sum(axis=0).astype(np.uint16)

        with np.errstate(all="ignore"):
            reducer = REDUCERS[self.method]
            radiance_out = reducer(radiance, axis=0).astype(np.float32)
            # mean LI weights observations equally; matches obs_count semantics
            li_out = np.nanmean(li, axis=0).astype(np.float32)

        below_min = obs_count < self.min_obs
        radiance_out[below_min] = np.nan
        li_out[below_min] = np.nan

        n_pixels_with_obs = int((obs_count > 0).sum())
        log.info(
            "%s %s: %d orbits → composite has %d/%d pixels with >=1 obs (max=%d)",
            self.sensor, period, len(records), n_pixels_with_obs,
            obs_count.size, int(obs_count.max()) if obs_count.size else 0,
        )

        self._write_float(outs["radiance"], radiance_out, ref_profile)
        self._write_float(outs["li"], li_out, ref_profile)
        self._write_count(outs["obs_count"], obs_count, ref_profile)

        return {
            "sensor": self.sensor,
            "period": period,
            "n_orbits": len(records),
            **outs,
        }

    # ---- IO helpers ---------------------------------------------------

    @staticmethod
    def _write_float(path: Path, arr: np.ndarray, ref_profile: dict) -> None:
        profile = ref_profile.copy()
        profile.update(
            dtype="float32",
            nodata=float("nan"),
            count=1,
            compress="deflate",
            tiled=True,
            driver="GTiff",
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
        tmp.replace(path)

    @staticmethod
    def _write_count(path: Path, arr: np.ndarray, ref_profile: dict) -> None:
        profile = ref_profile.copy()
        profile.update(
            dtype="uint16",
            nodata=0,
            count=1,
            compress="deflate",
            tiled=True,
            driver="GTiff",
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(arr.astype(np.uint16), 1)
        tmp.replace(path)
