"""Per-orbit cleaning: mask + warp to a common ROI grid.

Step 2 of the WB-LEN ingestion pipeline. Takes the (radiance, li, flag) triplet
produced by `harmonizer.ingest` for a single orbit and emits two layers on a
shared per-sensor ROI grid:

    radiance.tif  — masked + warped, float32 with NaN nodata
    li.tif        — masked + warped, float32 with NaN nodata

The flag layer is consumed (used to derive the mask) but not written out: a
downstream consumer can recover the valid-pixel mask via `np.isnan(radiance)`.

The mask combines:
  - the lunar bit, in one of three modes (zero | low | all)
  - any extra "mask if these bits are set" indices the caller passes
  - source NoData and obvious sentinel values

Each orbit is warped to a *per-sensor* common ROI grid (DMSP at 30 arc-sec,
VIIRS at 15 arc-sec by default), so the downstream compositor can stack frames
within a sensor without per-pair reprojection. Cross-sensor alignment is still
the harmonizer's job, exactly as in the legacy pipeline.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import Resampling, reproject

from harmonizer.constants import (
    SENSOR_CONFIGS,
    SENSOR_DMSP,
    SENSOR_VIIRS,
    SensorConfig,
)

log = logging.getLogger(__name__)

# Per-sensor native ROI-grid pixel size (degrees lat/lon, EPSG:4326).
NATIVE_PIXEL_SIZE_DEG = {
    SENSOR_DMSP: 30.0 / 3600.0,    # 30 arc-sec
    SENSOR_VIIRS: 15.0 / 3600.0,   # 15 arc-sec
}

LUNAR_MASK_MODES = ("zero", "low", "all")


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def decode_bit(flag: np.ndarray, bit: int) -> np.ndarray:
    """Return a boolean array marking pixels where `bit` is set in `flag`."""
    return ((flag.astype(np.int64) >> bit) & 1).astype(bool)


def make_lunar_mask(
    li: np.ndarray,
    flag: np.ndarray,
    cfg: SensorConfig,
    mode: str,
    low_thresh_lux: float,
) -> np.ndarray:
    """Boolean mask: True where the pixel passes the lunar-illum check.

    Modes:
        "zero" — keep only pixels where the sensor's zero-lunar bit is set
                 (the EOG composite convention).
        "low"  — keep pixels where 0 <= LI < low_thresh_lux.
        "all"  — keep every pixel (no lunar filter).
    """
    if mode == "all":
        return np.ones(flag.shape, dtype=bool)
    if mode == "zero":
        return decode_bit(flag, cfg.zero_lunar_bit)
    if mode == "low":
        return (li >= 0) & (li < low_thresh_lux) & np.isfinite(li)
    raise ValueError(f"unknown lunar_mask_mode: {mode!r} (expected one of {LUNAR_MASK_MODES})")


def make_validity_mask(
    radiance: np.ndarray, li: np.ndarray, cfg: SensorConfig
) -> np.ndarray:
    """Boolean: True where pixel has a real observation (not a fill sentinel).

    Two independent indicators:
      - LI within plausible physical range (>= 0, finite). Lunar illuminance
        is non-negative; both sensors use negative values (-1.0 / -999.3) as
        no-observation sentinels.
      - Radiance within the sensor's valid range. Catches DMSP's 255 fill and
        any extreme outliers without depending on a single sentinel value.
    """
    rad_f = radiance.astype(np.float32)
    rad_min, rad_max = cfg.radiance_range
    return (
        (li >= 0)
        & np.isfinite(li)
        & np.isfinite(rad_f)
        & (rad_f >= rad_min)
        & (rad_f <= rad_max)
    )


def make_extra_mask(flag: np.ndarray, mask_if_set: Iterable[int]) -> np.ndarray:
    """Boolean: True where NONE of the listed bits are set."""
    keep = np.ones(flag.shape, dtype=bool)
    for bit in mask_if_set:
        keep &= ~decode_bit(flag, bit)
    return keep


# ---------------------------------------------------------------------------
# Target grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetGrid:
    """Pixel-snapped ROI grid in EPSG:4326."""

    bounds: tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax) snapped
    width: int
    height: int
    pixel_size_deg: float

    @property
    def transform(self):
        xmin, ymin, xmax, ymax = self.bounds
        return transform_from_bounds(xmin, ymin, xmax, ymax, self.width, self.height)


def make_target_grid(
    roi_bbox: tuple[float, float, float, float],
    pixel_size_deg: float,
) -> TargetGrid:
    """Snap the ROI bbox out to whole-pixel boundaries at the given resolution."""
    xmin, ymin, xmax, ymax = roi_bbox
    xmin_s = math.floor(xmin / pixel_size_deg) * pixel_size_deg
    ymin_s = math.floor(ymin / pixel_size_deg) * pixel_size_deg
    xmax_s = math.ceil(xmax / pixel_size_deg) * pixel_size_deg
    ymax_s = math.ceil(ymax / pixel_size_deg) * pixel_size_deg
    width = int(round((xmax_s - xmin_s) / pixel_size_deg))
    height = int(round((ymax_s - ymin_s) / pixel_size_deg))
    return TargetGrid(
        bounds=(xmin_s, ymin_s, xmax_s, ymax_s),
        width=width,
        height=height,
        pixel_size_deg=pixel_size_deg,
    )


# ---------------------------------------------------------------------------
# OrbitPrep transformer
# ---------------------------------------------------------------------------

@dataclass
class OrbitPrep:
    """Mask + warp a single orbit triplet onto a common per-sensor ROI grid.

    Parameters
    ----------
    sensor : "dmsp" | "viirs_npp"
    roi_bbox : (xmin, ymin, xmax, ymax) in EPSG:4326
    dst_dir : output root; orbit outputs land in {dst_dir}/{sensor}/{period}/
    lunar_mask_mode : "zero" (default), "low", or "all"
    low_thresh_lux : LI threshold used in "low" mode (lux)
    extra_mask_if_set : flag bit indices that should mask a pixel out when set
    pixel_size_deg : output pixel size; defaults to sensor native
    """

    sensor: str
    roi_bbox: tuple[float, float, float, float]
    dst_dir: Path
    lunar_mask_mode: str = "zero"
    low_thresh_lux: float = 0.1
    extra_mask_if_set: tuple[int, ...] = field(default_factory=tuple)
    pixel_size_deg: Optional[float] = None
    _grid: TargetGrid = field(init=False)
    _cfg: SensorConfig = field(init=False)

    def __post_init__(self):
        if self.sensor not in SENSOR_CONFIGS:
            raise ValueError(f"unknown sensor: {self.sensor}")
        if self.lunar_mask_mode not in LUNAR_MASK_MODES:
            raise ValueError(
                f"lunar_mask_mode must be one of {LUNAR_MASK_MODES}, got {self.lunar_mask_mode!r}"
            )
        self._cfg = SENSOR_CONFIGS[self.sensor]
        if self.pixel_size_deg is None:
            self.pixel_size_deg = NATIVE_PIXEL_SIZE_DEG[self.sensor]
        self._grid = make_target_grid(self.roi_bbox, self.pixel_size_deg)
        self.dst_dir = Path(self.dst_dir)

    # ---- orchestration --------------------------------------------------

    def out_dir(self, period: str) -> Path:
        d = self.dst_dir / self.sensor / period
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out_paths(self, period: str, orbit_id: str) -> dict[str, Path]:
        d = self.out_dir(period)
        return {
            "radiance": d / f"{orbit_id}.radiance.tif",
            "li": d / f"{orbit_id}.li.tif",
        }

    def transform(self, record: dict) -> dict:
        """Process a single orbit record from `harmonizer.ingest.ingest()`.

        record schema: {"orbit": OrbitRef, "radiance": Path, "li": Path, "flag": Path}
        """
        orbit = record["orbit"]
        if orbit.sensor != self.sensor:
            raise ValueError(
                f"OrbitPrep configured for {self.sensor} but record sensor is {orbit.sensor}"
            )
        if any(record.get(k) is None for k in ("radiance", "li", "flag")):
            log.warning("orbit %s missing one or more layers; skipping", orbit.orbit_id)
            return {"orbit": orbit, "radiance": None, "li": None}

        period = orbit.datetime.strftime("%Y%m")
        outs = self.out_paths(period, orbit.orbit_id)
        if all(p.exists() and p.stat().st_size > 0 for p in outs.values()):
            return {"orbit": orbit, **outs}

        # Read source layers (all on same source grid — they're co-located).
        radiance_src, src_meta = self._read(record["radiance"])
        li_src, _ = self._read(record["li"])
        flag_src, _ = self._read(record["flag"])

        # Build mask in source space, apply, then warp.
        keep = make_lunar_mask(
            li_src, flag_src, self._cfg, self.lunar_mask_mode, self.low_thresh_lux
        )
        keep &= make_extra_mask(flag_src, self.extra_mask_if_set)
        keep &= make_validity_mask(radiance_src, li_src, self._cfg)

        radiance_masked = radiance_src.astype(np.float32, copy=True)
        radiance_masked[~keep] = np.nan
        li_masked = li_src.astype(np.float32, copy=True)
        li_masked[~keep] = np.nan

        kept_pct = 100.0 * keep.sum() / keep.size if keep.size else 0.0
        log.debug(
            "%s %s: kept %.1f%% of pixels after masking",
            self.sensor, orbit.orbit_id, kept_pct,
        )

        self._warp_and_write(
            radiance_masked, src_meta, outs["radiance"],
            resampling=Resampling.average,
        )
        self._warp_and_write(
            li_masked, src_meta, outs["li"],
            resampling=Resampling.average,
        )
        return {"orbit": orbit, **outs}

    # ---- IO helpers -----------------------------------------------------

    @staticmethod
    def _read(path: Path) -> tuple[np.ndarray, dict]:
        with rasterio.open(path) as src:
            arr = src.read(1)
            meta = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
            }
        return arr, meta

    def _warp_and_write(
        self,
        src_arr: np.ndarray,
        src_meta: dict,
        dst_path: Path,
        resampling: Resampling,
    ) -> None:
        dst = np.full((self._grid.height, self._grid.width), np.nan, dtype=np.float32)
        reproject(
            source=src_arr.astype(np.float32),
            destination=dst,
            src_transform=src_meta["transform"],
            src_crs=src_meta["crs"],
            src_nodata=np.nan,
            dst_transform=self._grid.transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=resampling,
        )
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": float("nan"),
            "width": self._grid.width,
            "height": self._grid.height,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": self._grid.transform,
            "compress": "deflate",
            "tiled": True,
        }
        tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **profile) as out:
            out.write(dst, 1)
        tmp.replace(dst_path)
