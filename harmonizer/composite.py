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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

import numpy as np
import rasterio
from rasterio.windows import Window

from harmonizer.constants import SENSOR_CONFIGS

log = logging.getLogger(__name__)

REDUCERS = {
    "median": np.nanmedian,
    "mean": np.nanmean,
}

# Row-block height for windowed compositing. Bounds the working set to
# ~(n_orbits × BLOCK_ROWS × width × 4B) per stack instead of the whole frame.
# 256 rows over a 3530-wide VIIRS grid × 200 orbits ≈ 720 MB per array — well
# under typical heap pressure thresholds while still amortizing IO overhead.
_COMPOSITE_BLOCK_ROWS = 256


@dataclass
class Compositor:
    """Reduce per-orbit rasters to per-period composites.

    Parameters
    ----------
    sensor : "dmsp" | "viirs_npp"
    dst_dir : output root; composites land in
              ``{dst_dir}/{sensor}/{roi_slug}/{period}/`` (see
              ``harmonizer.utils.roi_slug``). The slug must match the one
              from the ``OrbitPrep`` instance that produced the input
              records, otherwise composites would be written to a path
              divorced from their source grid identity.
    roi_slug : opaque cache namespace — pass ``OrbitPrep.roi_slug`` from the
               instance whose outputs you're compositing.
    method : "median" (default) or "mean"
    period_format : strftime pattern for the period key. Default "%Y%m"
                    (monthly). "%Y" for annual, "%Y%m%d" for daily.
    min_obs : minimum number of valid observations required per pixel; pixels
              below this threshold come out as NaN. Default 1.
    """

    sensor: str
    dst_dir: Path
    roi_slug: str
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
        for period in tqdm(sorted(groups), desc=f"{self.sensor} composite", unit="period"):
            results.append(self.composite_period(period, groups[period]))
        return results

    def composite_period(self, period: str, records: list[dict]) -> dict:
        """Reduce one period's worth of orbit records into a composite triplet.

        Processes one row-strip at a time so the working set stays bounded by
        ``_COMPOSITE_BLOCK_ROWS`` regardless of how many orbits the period
        contains or how big the ROI grid is. Numerically identical to the
        full-frame version since every reducer here is per-pixel.
        """
        out_dir = self.dst_dir / self.sensor / self.roi_slug / period
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = {
            "radiance": out_dir / "radiance.tif",
            "li": out_dir / "li.tif",
            "obs_count": out_dir / "obs_count.tif",
        }

        with rasterio.open(records[0]["radiance"]) as src0:
            ref_profile = src0.profile.copy()
            height, width = src0.height, src0.width

        rad_paths = [rec["radiance"] for rec in records]
        li_paths = [rec["li"] for rec in records]

        rad_profile = self._float_profile(ref_profile)
        li_profile = self._float_profile(ref_profile)
        count_profile = self._count_profile(ref_profile)

        rad_tmp = outs["radiance"].with_suffix(outs["radiance"].suffix + ".tmp")
        li_tmp = outs["li"].with_suffix(outs["li"].suffix + ".tmp")
        count_tmp = outs["obs_count"].with_suffix(outs["obs_count"].suffix + ".tmp")

        reducer = REDUCERS[self.method]
        n_pixels_with_obs = 0
        max_obs = 0

        with rasterio.open(rad_tmp, "w", **rad_profile) as rad_dst, \
             rasterio.open(li_tmp, "w", **li_profile) as li_dst, \
             rasterio.open(count_tmp, "w", **count_profile) as count_dst:
            for row_off in range(0, height, _COMPOSITE_BLOCK_ROWS):
                rows = min(_COMPOSITE_BLOCK_ROWS, height - row_off)
                window = Window(0, row_off, width, rows)

                rad_block = np.empty((len(records), rows, width), dtype=np.float32)
                li_block = np.empty((len(records), rows, width), dtype=np.float32)
                for i, (rp, lp) in enumerate(zip(rad_paths, li_paths)):
                    with rasterio.open(rp) as src:
                        src.read(1, window=window, out=rad_block[i])
                    with rasterio.open(lp) as src:
                        src.read(1, window=window, out=li_block[i])

                valid = np.isfinite(rad_block)
                obs_count_block = valid.sum(axis=0).astype(np.uint16)
                with np.errstate(all="ignore"):
                    rad_out_block = reducer(rad_block, axis=0).astype(np.float32)
                    li_out_block = np.nanmean(li_block, axis=0).astype(np.float32)

                below_min = obs_count_block < self.min_obs
                rad_out_block[below_min] = np.nan
                li_out_block[below_min] = np.nan

                rad_dst.write(rad_out_block, 1, window=window)
                li_dst.write(li_out_block, 1, window=window)
                count_dst.write(obs_count_block, 1, window=window)

                n_pixels_with_obs += int((obs_count_block > 0).sum())
                if obs_count_block.size:
                    max_obs = max(max_obs, int(obs_count_block.max()))

        rad_tmp.replace(outs["radiance"])
        li_tmp.replace(outs["li"])
        count_tmp.replace(outs["obs_count"])

        log.info(
            "%s %s: %d orbits → composite has %d/%d pixels with >=1 obs (max=%d)",
            self.sensor, period, len(records), n_pixels_with_obs,
            height * width, max_obs,
        )

        return {
            "sensor": self.sensor,
            "period": period,
            "n_orbits": len(records),
            **outs,
        }

    # ---- IO helpers ---------------------------------------------------

    @staticmethod
    def _float_profile(ref_profile: dict) -> dict:
        profile = ref_profile.copy()
        profile.update(
            dtype="float32",
            nodata=float("nan"),
            count=1,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            driver="GTiff",
        )
        return profile

    @staticmethod
    def _count_profile(ref_profile: dict) -> dict:
        profile = ref_profile.copy()
        profile.update(
            dtype="uint16",
            nodata=0,
            count=1,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            driver="GTiff",
        )
        return profile
