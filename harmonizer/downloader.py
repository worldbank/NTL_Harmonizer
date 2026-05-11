"""Deprecated. Kept as a shim so old imports don't crash.

The legacy pipeline downloaded ~115 GB of full-globe annual composites from
NOAA NGDC (DMSP-OLS) and EOG (VIIRS-DNB), then cropped them locally. The new
pipeline reads the WB-LEN STAC archive on S3 (`s3://globalnightlight/`) with
windowed COG access — no bulk download, no manual extraction.

Use `harmonizer.ingest.ingest(...)` and the rest of the pipeline (`OrbitPrep`,
`Compositor`, `calibrate_*`, `Harmonizer`) instead. See `harmonizer/main.py`
for the full wiring.
"""
import warnings


def _deprecated(name: str) -> None:
    warnings.warn(
        f"{name} has been removed. The new pipeline reads from "
        "s3://globalnightlight/ via harmonizer.ingest. "
        "See harmonizer/main.py for the new flow.",
        DeprecationWarning,
        stacklevel=2,
    )


def batch_download_DMSP(*args, **kwargs):
    _deprecated("batch_download_DMSP")
    raise RuntimeError("batch_download_DMSP is removed; use harmonizer.ingest instead")


def unzipDMSP(*args, **kwargs):
    _deprecated("unzipDMSP")
    raise RuntimeError("unzipDMSP is removed; the new pipeline reads COGs directly")


def unzipVIIRS(*args, **kwargs):
    _deprecated("unzipVIIRS")
    raise RuntimeError("unzipVIIRS is removed; the new pipeline reads COGs directly")
