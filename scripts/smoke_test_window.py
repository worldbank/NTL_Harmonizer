"""Focused timing smoke test for the windowed COG reader.

Skips the catalog walk (which we already know works) and goes straight to a
known-good VIIRS orbit to measure how long a windowed read actually takes
against the public bucket.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_test_window")

PARIS_BBOX = (2.0, 48.5, 3.0, 49.5)

# A specific VIIRS-NPP orbit known to overlap France on 2015-01-01.
RADIANCE_URL = (
    "https://globalnightlight.s3.amazonaws.com/201501/"
    "SVDNB_npp_d20150101_t0123164_e0128568_b16467_"
    "c20150101072856716086_noaa_ops.rade9.co.tif"
)


def time_read(url: str, dst: Path, gdal_env: dict) -> None:
    import rasterio
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import transform_bounds

    log.info("opening %s", url)
    t0 = time.time()
    with rasterio.Env(**gdal_env):
        with rasterio.open(url) as src:
            t1 = time.time()
            log.info("opened in %.2fs (size=%dx%d crs=%s)", t1 - t0, src.width, src.height, src.crs)
            xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", src.crs, *PARIS_BBOX)
            window = (
                from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                .round_offsets()
                .round_lengths()
                .intersection(Window(0, 0, src.width, src.height))
            )
            log.info("window: %s", window)
            data = src.read(1, window=window)
            t2 = time.time()
            log.info("read %s in %.2fs (sum=%.3f, min=%.3f, max=%.3f)", data.shape, t2 - t1, float(data.sum()), float(data.min()), float(data.max()))
            profile = src.profile.copy()
            profile.update(
                height=int(window.height),
                width=int(window.width),
                transform=src.window_transform(window),
                driver="GTiff",
                compress="deflate",
                tiled=True,
            )
            with rasterio.open(dst, "w", **profile) as out:
                out.write(data, 1)
    log.info("wrote %s (%.1f KB)", dst, dst.stat().st_size / 1024)


def main() -> int:
    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_VERSION": "2",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "paris_radiance.tif"
        time_read(RADIANCE_URL, dst, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
