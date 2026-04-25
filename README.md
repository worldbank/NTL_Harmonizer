# dmsp-viirs-intercalibration

Harmonizes DMSP-OLS and VIIRS-DNB nighttime lights into a coherent monthly
time series across platforms, ingesting nightly orbit COGs directly from the
World Bank [Light Every Night (LEN)](https://registry.opendata.aws/wb-light-every-night/)
S3 archive (`s3://globalnightlight/`).

Feedback welcome — please open an Issue or PR.

## What's new in this version

The previous pipeline required users to manually download ~115 GB of
compressed full-globe annual composites from NOAA NGDC and the EOG, then
locally extract and crop them per ROI. That step is gone.

The current pipeline:

1. Walks the LEN STAC catalog for orbits intersecting your ROI + date range.
2. Reads only the ROI window of each per-orbit COG (radiance, lunar
   illuminance, QA flag) directly from S3.
3. Masks per-orbit pixels by lunar bit / QA flags / sensor validity range.
4. Composites nightly orbits to per-period (default monthly) rasters on a
   per-sensor common ROI grid.
5. Applies the legacy DMSP intercalibration (Li et al. 2017 stepwise) and
   VIIRS preprocessing (damper / Gaussian / log1p) at the period level.
6. Fits the Harmonizer on the training year's monthly stack and applies it
   to every available VIIRS period.

This means: no manual downloads, monthly granularity by default (instead of
annual), and DMSP coverage extended through 2017 (the LEN nightly archive
goes further than the EOG annual composites).

## File structure

- `harmonizer/`: main package
  - `constants.py` — WB-LEN bucket info, sensor configs (NoData, lunar bit, valid radiance range, layer name resolvers).
  - `ingest.py` — STAC walker + windowed COG reader. Replaces the old `downloader.py`.
  - `composite.py` — per-period mosaicking / median reducer.
  - `calibrate.py` — per-period batch wrappers around DMSPstepwise + VIIRSprep.
  - `transformers/`
    - `orbitprep.py` — per-orbit cleaning: mask + warp to common ROI grid.
    - `dmspcalibrate.py` — DMSP-OLS Li 2017 intercalibration.
    - `viirsprep.py` — VIIRS-DNB damper / Gaussian / log1p preprocessing.
    - `harmonize.py` — monthly-cadence Harmonizer (XGB / curve / no-fit estimators).
    - `gbm.py` — XGBoost wrapper.
    - `curve.py` — polynomial curve / no-fit estimators.
  - `diagnostics.py` — per-training-period scatter/hist + monthly time series.
  - `main.py` — top-level CLI entry point.
  - `config.py` — paths, defaults, ROI selection.
  - `utils.py` — shared helpers including `roi_bbox_from_path`.
- `roifiles/` — example shapefiles (France, Italy, Spain, USA, etc.)
- `scripts/` — smoke tests for each pipeline stage.

## Hardware requirements

- Standard laptop with 16 GB RAM is plenty for country-scale monthly
  composites at 30 arc-sec.
- ~5–20 GB free for the local cache (`data/cache/...`) for a single trial;
  varies with ROI and date range.
- Running on EC2 in `us-east-1` avoids cross-region S3 egress for very large
  ROIs.

## Setup

### Conda

```sh
conda env create -f environment.yml
conda activate ntl_harmonizer
conda develop .
```

### Pip

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GDAL must be installed system-wide for `rasterio` (and optionally `fiona`,
used to read shapefiles). On macOS use `brew install gdal`. On Ubuntu use
`apt install libgdal-dev`. Confirm with `gdalinfo --version`.

Set `NLT` to the parent directory containing this repo (e.g. `export NLT=$HOME`).

## Configuration

`harmonizer/config.py` sets the defaults. The two settings users typically
change:

- `roipath` — path to a shapefile in `roifiles/`, **or** an
  `"xmin,ymin,xmax,ymax"` CSV in EPSG:4326 if you want to skip the shapefile.
- `START_DATE` / `END_DATE` — defaults to 1992–2013, the canonical DMSP
  era. Extend `END_DATE` and add later years to `DMSP_PREFERRED_SATS` if you
  want to use the full DMSP-F18 archive (through 2017).

Other knobs (mostly leave alone):

- `PERIOD_FORMAT` — `"%Y%m"` (monthly, default) / `"%Y"` (annual) / `"%Y%m%d"` (daily).
- `LUNAR_MASK_MODE` — `"zero"` (EOG composite convention, default), `"low"`, `"all"`.
- `TRAIN_YEAR` — year used to fit the Harmonizer (default 2013).
- `DOWNSAMPLEVIIRS` — True (default) ⇒ all output at DMSP 30 arc-sec.
- `SAMPLEMETHOD` — rasterio.warp resampling kernel name.

Cache paths land under `data/cache/` (already in `.gitignore` via the
existing `data/` rule). Output rasters land at
`output/{trialname}/{period}/radiance.tif`. Diagnostic plots and metrics land
at `results/{trialname}/`.

## Running

```sh
python -m harmonizer.main -n my_trial
```

Useful flags:

```sh
python -m harmonizer.main \
    -n france_2010s \
    --roi roifiles/gadm36_FRA_shp/gadm36_FRA_0.shp \
    --start 2012-04-01 --end 2013-12-31 \
    --train-year 2013 \
    --lunar-mode zero
```

Or with a bbox shorthand:

```sh
python -m harmonizer.main -n paris --roi "2.0,48.5,3.0,49.5"
```

For partial reruns, the cache under `data/cache/` is keyed by
sensor+period+orbit, so re-running with a wider date range only fetches new
orbits.

## Outputs

For trial `<name>`:

- `output/<name>/{period}/radiance.tif` — harmonized VIIRS rasters (one per
  inference period).
- `artifacts/<name>_harmonizer.dill` — the trained Harmonizer (load with
  `harmonizer.transformers.harmonize.load_obj`).
- `results/<name>/`
  - `{period}_scatter.png`, `{period}_hist.png` — for each training period.
  - `harmonized_ts_mean.png`, `..._median.png`, `..._sum.png` — full
    timeseries plots, DMSP intercalibrated vs. VIIRS harmonized.
  - `training_metrics.txt` — per-training-period RMSD + Spearman R.

The compositor also leaves intermediate per-period rasters in the cache —
useful for ablations, debugging, and direct downstream use without rerunning.

## DMSP coverage notes

Per the legacy `DMSP_PREFERRED_SATS` mapping (Li et al. 2017), each year picks
one DMSP satellite for intercalibration. The mapping covers F10/F12 (no
published coefficients — falls back to clipping) through F18 2010–2013. To
extend coverage past 2013, add F18 entries for 2014–2017 and re-run. F10/F12
years currently emit clipped-only output (matches the legacy behavior).

## Going further

- The `lunar mask mode` and `mean LI` per-period rasters open up
  experiments that the legacy annual-composite pipeline couldn't do — e.g.
  lunar-stratified harmonization, or using LI as a model feature. See
  `NTL_Harmonizer_Clay_Integration_Plan.md` for the longer roadmap.
- Smoke tests in `scripts/` exercise each pipeline stage in isolation. Useful
  starting points for understanding the data and the modules.

## Background

For an introduction to DMSP-OLS and VIIRS-DNB, see the World Bank
[Open Nighttime Lights tutorial](https://worldbank.github.io/OpenNightLights/tutorials/mod1_2_introduction_to_nighttime_light_data.html).
