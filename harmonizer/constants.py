"""WB-LEN archive constants and per-sensor layer conventions.

The World Bank "Light Every Night" archive (s3://globalnightlight) ships, per
orbit segment, several co-located COG layers. We need three of them:

    radiance : the visible-band signal we harmonize
    li       : per-pixel lunar illuminance in lux
    flag     : per-pixel bitfield with QA flags (zero-lunar-illum, cloud, etc.)

The two sensors use different filename conventions and different NoData /
bit-index conventions, so this module centralizes the mapping. Anything that
hardcodes a NoData value or a bit position should pull it from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

S3_BUCKET = "globalnightlight"
S3_REGION = "us-east-1"
S3_HTTPS_BASE = f"https://{S3_BUCKET}.s3.amazonaws.com"

ROOT_CATALOG_URL = f"{S3_HTTPS_BASE}/catalog.json"
DMSP_CATALOG_URL = f"{S3_HTTPS_BASE}/DMSP_catalog.json"
VIIRS_CATALOG_URL = f"{S3_HTTPS_BASE}/VIIRS_npp_catalog.json"

SENSOR_DMSP = "dmsp"
SENSOR_VIIRS = "viirs_npp"
SENSORS = (SENSOR_DMSP, SENSOR_VIIRS)

# NoData values for the LI layer (lunar illuminance, lux)
DMSP_LI_NODATA = -1.0
VIIRS_LI_NODATA = -999.3

# Lunar-illumination bit positions in the QA flag layer (LSB = 0).
# Set ⇒ that pixel was acquired under effectively zero moonlight (EOG composite
# convention is to keep only these pixels).
OLS_ZERO_LUNAR_ILLUM_BIT = 11
VIIRS_ZERO_LUNAR_ILLUM_BIT = 5

# DMSP filename: F16200501010404.night.OIS.{vis|li|flag|tir|samples}.co.tif
# Same orbit prefix across all six layers, so substitution is direct.
_DMSP_RADIANCE_RE = re.compile(r"\.OIS\.vis\.co\.tif$")


def _dmsp_li_from_radiance(url: str) -> str:
    if not _DMSP_RADIANCE_RE.search(url):
        raise ValueError(f"Not a DMSP radiance URL: {url}")
    return _DMSP_RADIANCE_RE.sub(".OIS.li.co.tif", url)


def _dmsp_flag_from_radiance(url: str) -> str:
    if not _DMSP_RADIANCE_RE.search(url):
        raise ValueError(f"Not a DMSP radiance URL: {url}")
    return _DMSP_RADIANCE_RE.sub(".OIS.flag.co.tif", url)


# VIIRS filename anatomy:
#   radiance : SVDNB_npp_d{date}_t{...}_e{...}_b{orbit}_c{ts1}_noaa_ops.rade9.co.tif
#   li       : GDNBO_npp_d{date}_t{...}_e{...}_b{orbit}_c{ts2}_noaa_ops.li.co.tif
#   flag     : npp_d{date}_t{...}_e{...}_b{orbit}.vflag.co.tif
# Note: the SVDNB and GDNBO files have *different* c{processing_timestamp}
# suffixes for the same physical orbit segment, so we cannot derive the LI URL
# by pure string substitution. Instead we extract the orbit identifier
# `npp_d…_t…_e…_b…` and search the bucket for the matching GDNBO file.
# The flag file uses just the orbit identifier (no `c…_noaa_ops` suffix), so
# it can be derived directly.
_VIIRS_ORBIT_RE = re.compile(
    r"(?P<orbit>npp_d\d+_t\d+_e\d+_b\d+)"
)
_VIIRS_RADIANCE_RE = re.compile(
    r"SVDNB_(?P<orbit>npp_d\d+_t\d+_e\d+_b\d+)_c\d+_noaa_ops\.rade9\.co\.tif$"
)


def viirs_orbit_key(url_or_filename: str) -> str:
    """Return the sensor-agnostic orbit identifier for a VIIRS filename.

    e.g. ".../SVDNB_npp_d20150101_t0003355_e0009159_b16466_c2015..._noaa_ops.rade9.co.tif"
         -> "npp_d20150101_t0003355_e0009159_b16466"
    """
    m = _VIIRS_ORBIT_RE.search(url_or_filename)
    if not m:
        raise ValueError(f"No VIIRS orbit identifier in: {url_or_filename}")
    return m.group("orbit")


def _viirs_flag_from_radiance(url: str) -> str:
    """Derive the VIIRS flag file URL from the radiance URL.

    The flag file lives in the same prefix and uses just the orbit key:
        npp_d20150101_t0003355_e0009159_b16466.vflag.co.tif
    """
    m = _VIIRS_RADIANCE_RE.search(url)
    if not m:
        raise ValueError(f"Not a VIIRS radiance URL: {url}")
    orbit = m.group("orbit")
    prefix = url[: m.start()]
    return f"{prefix}{orbit}.vflag.co.tif"


# `li_resolver` is a placeholder: VIIRS LI requires a bucket lookup (different
# c{ts}), so the actual resolution happens in ingest.py with a list-by-prefix
# call. We expose just the orbit-key extractor here.
def _viirs_li_from_radiance(url: str) -> str:  # pragma: no cover
    raise NotImplementedError(
        "VIIRS LI URL must be resolved by S3 prefix lookup (different "
        "processing timestamp). Use ingest.resolve_viirs_li_url()."
    )


# Per-sensor inclusive valid range for the radiance band; pixels outside this
# range are assumed to be fill / nodata. Empirical observation:
#   DMSP OLS vis : uint8, valid 0–63, fill = 255
#   VIIRS rade9  : float32, valid >= 0; sentinel for "no observation" co-occurs
#                  with li == -999.3, which the LI-validity check already rejects
DMSP_RADIANCE_RANGE = (0.0, 63.0)
VIIRS_RADIANCE_RANGE = (0.0, 1.0e9)


@dataclass(frozen=True)
class SensorConfig:
    name: str
    catalog_url: str
    li_nodata: float
    zero_lunar_bit: int
    radiance_range: tuple[float, float]
    li_from_radiance: Callable[[str], str]
    flag_from_radiance: Callable[[str], str]


SENSOR_CONFIGS = {
    SENSOR_DMSP: SensorConfig(
        name=SENSOR_DMSP,
        catalog_url=DMSP_CATALOG_URL,
        li_nodata=DMSP_LI_NODATA,
        zero_lunar_bit=OLS_ZERO_LUNAR_ILLUM_BIT,
        radiance_range=DMSP_RADIANCE_RANGE,
        li_from_radiance=_dmsp_li_from_radiance,
        flag_from_radiance=_dmsp_flag_from_radiance,
    ),
    SENSOR_VIIRS: SensorConfig(
        name=SENSOR_VIIRS,
        catalog_url=VIIRS_CATALOG_URL,
        li_nodata=VIIRS_LI_NODATA,
        zero_lunar_bit=VIIRS_ZERO_LUNAR_ILLUM_BIT,
        radiance_range=VIIRS_RADIANCE_RANGE,
        li_from_radiance=_viirs_li_from_radiance,
        flag_from_radiance=_viirs_flag_from_radiance,
    ),
}
