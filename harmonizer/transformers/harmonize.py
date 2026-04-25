"""DMSP–VIIRS harmonizer fit/applied on per-period composites.

Step 5 of the WB-LEN ingestion pipeline. The math (XGB / curve / no-fit
estimators, polynomial / shifted-neighbor / index-grid features) is unchanged
from the legacy annual-composite version. What's new:

  - Training pulls every paired (DMSP, VIIRS) composite in `train_periods`
    instead of a single 2013 annual frame, and concatenates the per-period
    pixel matrices. With the default 2013 monthly stack this gives ~12× more
    training pixels per ROI plus modest within-year variance.

  - Inputs are read from the per-period directories produced by step 4
    (`{dmsp_period_dir}/{period}/radiance.tif`,
     `{viirs_period_dir}/{period}/radiance.tif`). Outputs land at
    `{output_dir}/{period}/radiance.tif`.

  - Resampling between sensor grids uses in-memory `rasterio.warp.reproject`
    instead of an external gdalwarp call + staging file.

  - Composites carry NaN nodata (real masked pixels). Training filters NaN
    rows; inference preserves NaN through the estimator and writes NaN-nodata
    output rasters.

  - The two resolution options from the legacy class still apply:
      * downsampleVIIRS=True (default): everything output at DMSP 30 arc-sec
      * downsampleVIIRS=False: training upsamples DMSP to VIIRS 15 arc-sec;
        inference applies on VIIRS native and emits 15 arc-sec output.
    See the legacy docstring (preserved below in `_RESOLUTION_NOTES`) for the
    rationale.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import dill
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from harmonizer.utils import clip_arr

log = logging.getLogger(__name__)

_RESAMPLING_BY_NAME = {
    "average": Resampling.average,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "nearest": Resampling.nearest,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
}


def _resampling_from_name(name: str) -> Resampling:
    if name not in _RESAMPLING_BY_NAME:
        raise ValueError(
            f"unknown resampling method: {name!r} (have {sorted(_RESAMPLING_BY_NAME)})"
        )
    return _RESAMPLING_BY_NAME[name]


class Harmonizer:
    """Fit and apply the DMSP–VIIRS harmonization model on per-period composites.

    Parameters
    ----------
    dmsp_period_dir : Path
        Root of calibrated DMSP composites; expects {period}/radiance.tif
        underneath (the layout produced by `harmonizer.calibrate`).
    viirs_period_dir : Path
        Root of prepped VIIRS composites; same layout.
    output_dir : Path
        Where harmonized VIIRS rasters are written, one per inference period.
    train_periods : Iterable[str]
        Period keys to train on. The default 12 monthly keys for 2013 build a
        12-frame stack; any subset works (annual cadence is also fine).
    downsampleVIIRS : bool
        See `_RESOLUTION_NOTES`. True ⇒ output at DMSP 30 arc-sec.
    samplemethod : str
        rasterio.warp resampling kernel name (e.g. "average").
    polyX, shift, idX, epochs, est, opath
        Same semantics as the legacy class; see `Xy_transform`.
    """

    def __init__(
        self,
        dmsp_period_dir,
        viirs_period_dir,
        output_dir,
        train_periods: Iterable[str],
        downsampleVIIRS: bool = True,
        samplemethod: str = "average",
        polyX: bool = True,
        shift: bool = False,
        idX: bool = False,
        epochs: int = 100,
        est=None,
        opath=None,
    ):
        self.dmsp_period_dir = Path(dmsp_period_dir)
        self.viirs_period_dir = Path(viirs_period_dir)
        self.output_dir = Path(output_dir)
        self.train_periods = list(train_periods)
        self.downsampleVIIRS = downsampleVIIRS
        self.samplemethod = samplemethod
        self.polyX = polyX
        self.shift = shift
        self.idX = idX
        self.epochs = epochs
        self.est = est
        self.opath = opath

        if not self.train_periods:
            raise ValueError("train_periods must be non-empty")
        if est is None:
            raise ValueError("est is required (e.g. XGB(), CurveFit(), NoFit())")

    # ---- IO -------------------------------------------------------------

    def _radiance_path(self, root: Path, period: str) -> Path:
        return root / period / "radiance.tif"

    def load_arr(self, srcpath) -> np.ndarray:
        with rasterio.open(srcpath) as src:
            return src.read(1).astype(np.float32)

    def _reproject_to_ref(
        self, srcpath: Path, refpath: Path, resampling: Resampling
    ) -> tuple[np.ndarray, dict]:
        """Reproject `srcpath`'s band 1 onto `refpath`'s grid, return (array, ref_profile)."""
        with rasterio.open(refpath) as ref:
            ref_profile = ref.profile.copy()
            dst = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
            with rasterio.open(srcpath) as src:
                reproject(
                    source=src.read(1).astype(np.float32),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata if src.nodata is not None else np.nan,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
        return dst, ref_profile

    def _read_aligned_pair(self, period: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (viirs_arr, dmsp_arr, ref_profile) on a shared grid for one period."""
        dmsp_path = self._radiance_path(self.dmsp_period_dir, period)
        viirs_path = self._radiance_path(self.viirs_period_dir, period)
        if not dmsp_path.exists():
            raise FileNotFoundError(dmsp_path)
        if not viirs_path.exists():
            raise FileNotFoundError(viirs_path)
        method = _resampling_from_name(self.samplemethod)
        if self.downsampleVIIRS:
            viirs_arr, ref_profile = self._reproject_to_ref(viirs_path, dmsp_path, method)
            dmsp_arr = self.load_arr(dmsp_path)
        else:
            dmsp_arr, ref_profile = self._reproject_to_ref(dmsp_path, viirs_path, method)
            viirs_arr = self.load_arr(viirs_path)
        return viirs_arr, dmsp_arr, ref_profile

    # ---- features (unchanged from legacy, kept verbatim) ----------------

    def Xy_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.expand_dims(X, axis=0)
        features = [X]
        if self.polyX:
            features = [X, X ** 2]
        if self.shift:
            Xup = X.copy()
            Xup[0, :-1, :] = X[0, 1:, :]
            Xdn = X.copy()
            Xdn[0, 1:, :] = X[0, :-1, :]
            Xl = X.copy()
            Xl[0, :, :-1] = X[0, :, 1:]
            Xr = X.copy()
            Xr[0, :, 1:] = X[0, :, :-1]

            Xup2 = X.copy()
            Xup2[0, :-2, :] = X[0, 2:, :]
            Xdn2 = X.copy()
            Xdn2[0, 2:, :] = X[0, :-2, :]
            Xl2 = X.copy()
            Xl2[0, :, :-2] = X[0, :, 2:]
            Xr2 = X.copy()
            Xr2[0, :, 2:] = X[0, :, :-2]
            features = features + [Xup, Xdn, Xl, Xr, Xup2, Xdn2, Xl2, Xr2]

        if self.idX:
            idgrid = np.meshgrid(range(0, X.shape[-1]), range(0, X.shape[-2]))
            xgrid = np.expand_dims(idgrid[0], axis=0)
            ygrid = np.expand_dims(idgrid[1], axis=0)
            features = features + [xgrid, ygrid]
        X = np.concatenate(features, axis=0)
        X = X.reshape(X.shape[0], -1).T
        if y is not None:
            y = y.flatten()
        return X, y

    # ---- train ----------------------------------------------------------

    def train_prep(self) -> tuple[np.ndarray, np.ndarray]:
        Xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        used: list[str] = []
        for period in self.train_periods:
            try:
                viirs_arr, dmsp_arr, _ = self._read_aligned_pair(period)
            except FileNotFoundError as e:
                log.warning("training period %s missing (%s); skipping", period, e)
                continue
            X, y = self.Xy_transform(viirs_arr, dmsp_arr)
            valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if not valid.any():
                log.warning("training period %s has no valid pixels", period)
                continue
            Xs.append(X[valid])
            ys.append(y[valid])
            used.append(period)
        if not Xs:
            raise RuntimeError(
                f"no usable training periods found among {self.train_periods}"
            )
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        log.info(
            "training set assembled: %d periods, %d pixels (used: %s)",
            len(used), X.shape[0], used,
        )
        return X, y

    def fit(self) -> None:
        X, y = self.train_prep()
        self.est.fit(X, y, self.epochs)

    # ---- inference ------------------------------------------------------

    def process(self, X: np.ndarray) -> np.ndarray:
        shp = X.shape
        X_features, _ = self.Xy_transform(X)
        feature_valid = np.isfinite(X_features).all(axis=1)
        Y = np.full(X_features.shape[0], np.nan, dtype=np.float32)
        if feature_valid.any():
            Y[feature_valid] = self.est.predict(X_features[feature_valid])
            Y[feature_valid] = clip_arr(Y[feature_valid], floor=0.0, ceiling=63.0)
        return Y.reshape(shp).astype(np.float32)

    def transform(self, period: str) -> Optional[Path]:
        """Apply the fit harmonizer to one VIIRS period; return the output path."""
        viirs_path = self._radiance_path(self.viirs_period_dir, period)
        if not viirs_path.exists():
            log.warning("no VIIRS frame for period %s", period)
            return None

        method = _resampling_from_name(self.samplemethod)
        if self.downsampleVIIRS:
            # Resample VIIRS to a DMSP-grid reference (any training period works).
            ref_path = self._radiance_path(self.dmsp_period_dir, self.train_periods[0])
            X, profile = self._reproject_to_ref(viirs_path, ref_path, method)
        else:
            X = self.load_arr(viirs_path)
            with rasterio.open(viirs_path) as src:
                profile = src.profile.copy()

        Y = self.process(X)
        out_path = self.output_dir / period / "radiance.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(
            dtype="float32",
            nodata=float("nan"),
            count=1,
            compress="deflate",
            tiled=True,
            driver="GTiff",
        )
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(Y, 1)
        tmp.replace(out_path)
        return out_path


def save_obj(obj, opath):
    with open(opath, "wb") as dst:
        dill.dump(obj, dst)
    print(f"model artifact saved at {str(opath)}")


def load_obj(srcpath):
    with open(srcpath, "rb") as src:
        return dill.load(src)


_RESOLUTION_NOTES = """
Two choices regarding VIIRS-DNB spatial resolution:

Option 1 (default, downsampleVIIRS=True): VIIRS-DNB is resampled to 30 arc-sec
to match DMSP-OLS pixel-by-pixel for fitting. All inference VIIRS frames are
also resampled before the model is applied. Final time series is uniformly at
30 arc-sec.

Option 2 (downsampleVIIRS=False): The DMSP-OLS training frames are *upsampled*
to 15 arc-sec to match VIIRS pixel-by-pixel. Inference applies the model at
VIIRS native 15 arc-sec. Output mixes 30 arc-sec DMSP and 15 arc-sec VIIRS;
cross-platform aggregations require per-pixel standardization.
"""
