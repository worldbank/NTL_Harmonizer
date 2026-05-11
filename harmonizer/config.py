"""Pipeline configuration.

The new pipeline is WB-LEN native: nothing is downloaded ahead of time. ROI +
date range drive STAC-windowed reads against the public bucket, and every
intermediate stage caches its output under `data/cache/...`.

Legacy paths (DMSP_IN, VIIRS_IN, DMSP_CLIP, VIIRS_CLIP, *_TMP, *_URLS) are no
longer needed and have been retired. If you have notebooks or scripts that
imported them, see `harmonizer.downloader` for the deprecation note.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#########################
# ROI FILE
#########################
# Either a shapefile path, or an "xmin,ymin,xmax,ymax" CSV (EPSG:4326 lon/lat).
# Examples:
#   roipath = "roifiles/gadm36_FRA_shp/gadm36_FRA_0.shp"
#   roipath = "2.0,48.5,3.0,49.5"   # tight Paris bbox
roipath = "roifiles/gadm36_FRA_shp/gadm36_FRA_0.shp"

###################################
# DATE RANGE & PERIOD CADENCE
###################################
# Default spans both platforms: DMSP (1992–2013) + VIIRS (2012–present).
START_DATE = datetime(1992, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

# strftime pattern for the period key. "%Y%m" = monthly (default),
# "%Y" = annual, "%Y%m%d" = daily.
PERIOD_FORMAT = "%Y%m"

# Year used to fit the harmonizer (DMSP and VIIRS overlap in 2012–2013).
TRAIN_YEAR = 2013

# Lunar-illumination masking mode used in OrbitPrep. Matches the EOG zero-LI
# composite convention by default. Valid: "zero" | "low" | "all".
LUNAR_MASK_MODE = "zero"

###################################
# OPTIONAL CHANGES
###################################

# rasterio.warp resampling kernel (used for cross-sensor alignment in the
# Harmonizer). Options: average, bilinear, cubic, nearest, mode, max, min,
# med, q1, q3.
SAMPLEMETHOD = "average"

# True ⇒ output time series is uniformly at DMSP 30 arc-sec.
# False ⇒ DMSP training is upsampled to VIIRS 15 arc-sec; VIIRS retains 15 arc-sec
# resolution at inference. See harmonize.py docstring for full discussion.
DOWNSAMPLEVIIRS = True

###################################
# PATHS — usually no need to change
###################################
ROIPATH = Path(ROOT, roipath) if not roipath.count(",") == 3 else roipath
DATA = Path(ROOT, "data")
DATA.mkdir(exist_ok=True)

# Cache root for the new pipeline. Each stage gets its own subdir so reruns
# are incremental — windowed COGs and cleaned/composited rasters are reused.
CACHE = Path(DATA, "cache")
CACHE.mkdir(exist_ok=True)
INGEST_CACHE = Path(CACHE, "ingest")        # windowed orbit triplets (radiance/li/flag)
PREP_DIR = Path(CACHE, "orbitprep")          # per-orbit masked + warped clips
COMPOSITE_DIR = Path(CACHE, "composite")     # per-period (radiance, li, obs_count) composites
CALIB_DIR = Path(CACHE, "calibrated")        # post-DMSPstepwise per-period DMSP rasters
VIIRS_PREP_DIR = Path(CACHE, "viirs_prepped")  # post-VIIRSprep per-period VIIRS rasters
for d in (INGEST_CACHE, PREP_DIR, COMPOSITE_DIR, CALIB_DIR, VIIRS_PREP_DIR):
    d.mkdir(parents=True, exist_ok=True)

ARTIFACTS = Path(ROOT, "artifacts")
ARTIFACTS.mkdir(exist_ok=True)
OUTPUT = Path(ROOT, "output")
OUTPUT.mkdir(exist_ok=True)
RESULTS = Path(ROOT, "results")
RESULTS.mkdir(exist_ok=True)


# Preferred satellite per year for DMSP intercalibration (Li et al. 2017).
# The ingest layer uses this to filter STAC walks to single-satellite catalogs
# per year, so monthly composites are never satellite-mixed.
DMSP_PREFERRED_SATS = {
    "F10": ["1992", "1993", "1994"],
    "F12": ["1995", "1996"],
    "F14": ["1997", "1998", "1999", "2000", "2001", "2002", "2003"],
    "F16": ["2004", "2005", "2006", "2007", "2008", "2009"],
    "F18": ["2010", "2011", "2012", "2013"],
}
