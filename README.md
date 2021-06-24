# dmsp-viirs-intercalibration [beta]
Repo for project to harmonize DMSP-OLS and VIIRS-DNB nighttime lights to form a coherent time series across platforms.

A tutorial is being developed to provide usage instructions and examples.

Feedback on this is welcome and encouraged. Please feel free to create branches and add Issues within this repo.

## File structure
- `files/`: contains reference files (can be ignored)
- `harmonizer/`: main script directory
  - `transformers/`: contains modules for processing/transformations
    - `dmspcalibrate.py`: class of functions for DMSP-OLS intercalibration
    - `gbm.py`: class defining the harmonizer model: XGBoost (a gradient boosting machine) and default hyper-parameters
    - `harmonize.py`: class of functions for the final DMSP-VIIRS harmonization
    - `viirsprep.py`: class of functions for the VIIRS-DNB pre-processing steps
  - `config.py`: contains key global variables (see below)
  - `diagnostics.py`: set of functions for creating the result metrics and plots
  - `main.py`: main executable file
  - `plots.py`: plotting functions
  - `utils.py`: various utility functions
- `.gitignore`
- `README.md`
- `environment.yml`: conda dependency management
- `requirements.txt`: pip dependency management
- `setup.py`: sets up modules as custom Python packages (see below)

## Hardware requirements
- At minimum a standard laptop w/ 16Gb RAM, 4 CPUs, about ~115Gb in available storage space.
  A cloud-based approach is being explored.

## Getting started
### Step 1. Clone this repo locally (this is root)

### Step 2. Setup Python env
- Note: Docker setup and instructions forthcoming...

This package uses GDAL, a powerful open source geospatial software package (https://gdal.org/).
If you work with geospatial data often, you'll be familiar and have this on your machine system-wide (confirm this in CLI with this command: `$ gdalinfo --version`)

If you don't and are unfamiliar with GDAL, it's highly recommended you use `conda` (option 1 below).

#### Option 1: conda
**This is recommended if you're inexperienced with managing GIS packages with Python/pip, especially if you're using Windows.**
1. Set up new conda env using .yml file (e.g. from CLI in root directory: `$ conda env create -f environment.yml`)
  - Note the default name of the env is in the environment.yml file: `ntl_harmonizer`
2. Activate new env (e.g. from CLI: `$ conda activate ntl_harmonizer`)
3. Add this repo path so you can use the custom modules and packages. In your activated conda env in CLI from root: `$ conda develop .`

#### Option 2: pip and venv:
1. Create your virtual env (e.g. from CLI in root directory: `$ python -m venv venv`)
2. Activate your venv (e.g. from CLI: `$ source venv/bin/activate`) note for Windows directory switches are: \
3. You'll need to install the GDAL library if you haven't already. 
    - For Linux (Ubuntu disto) this gives guidance for installing GDAL and Python bindings: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
    - For Mac, the distribution manager `homebrew` is the way to go: https://trac.osgeo.org/gdal/wiki/BuildingOnMac
    - For Windows this is a trusted source, search for "GDAL" and find the architecture (e.g. amd64) and Python version (e.g. cp39 for Python 3.9) that fits your needs: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
        - For Windows installation you also may need to search this repo for binaries related to Fiona and rasterio
    - Confirm your installation in CLI: `$ gdalinfo --version`
4. Install dependencies: (e.g. from CLI in your activated env: `$ pip install -r requirements.txt`)
5. Install custom packages: (e.g. from CLI in your activated env: `$ pip install -e .`)

You should now have an active environment (either pip/venv or conda) with all required dependencies.

### Step 3. Set config variables
1. From CLI at root directory, enter: `$ python -m harmonizer.config`. This will create empty directories for the data later.
    - Note: the `config.py` file contains a variable `roipath` that you'll update with your ROI shapefile in a moment (see below)
    - Dont change any of the other variables to use default settings.

### Step 4. Download data
The current package leverages existing DMSP-OLS and VIIRS-DNB annual composites (see "Notes on future versions and data" below).
**Downloading and extracting these composites locally takes time and storage space (~25Gb for uncompressed DMSP files and ~85Gb for uncompressed VIIRS files),
but is temporary until COG data in the WB-LEN are cataloged and available.**

The data source links also contain technical specifications.

#### 4A: Download DMSP-OLS data
DMSP-OLS source (Version 4 DMSP-OLS Nighttime Lights Time Series). https://www.ngdc.noaa.gov/eog/dmsp/downloadV4composites.html

There are two options for this: 1) automatically downloading and 2) manually downloading

- Option 1: Auto downloading
For security reasons, it's unsafe to download and extract files unless you've verified the source. As such, if you are not on your own computer (i.e. a network computer)
and/or not a root user, your system admins may not allow this autodownload function to work and you'll have to manually download.

From CLI in root directory:`$ python -m harmonizer.downloader -dmsp`

This will download all the necessary DMSP-OLS files automatically and save them in the location set by the variables `DMSP_IN` in the `config.py` file.
Depending on internet bandwidth speed this will take up to 15-20 minutes.

- Option 2: Manual downloading
  - Open the text file: `files/noaa_dmsp_annual_urls.txt`, which has a list of 32 URLs for each composite
  - One at a time, copy and paste each url in a browser tab (such as Chrome) and the download should begin automatically, saving the file in your default "download" location
  - Move these files to the folder in this directory: `data/dmspcomps/` that was created when you initialized the `config.py` file in step 3. (Or you can set this to be your browser's download location.)

#### 4B: Download VIIRS-DNB data
VIIRS-DNB source (Earth Observation Group's Version 2 annual composites: https://eogdata.mines.edu/products/vnl/#annual_v2)

EOG requires a user account to download data. This is free and easy to do but makes automatic downloading more complex.
Since this is a temporary process (see the "Notes on future versions and data" below) no automated download process was created.

  - Open the text file: `files/eog_viirs_annualv2_urls.txt`, which has a list of URLs for each composite
  - One at a time, copy and paste each url in a browser tab (such as Chrome) and the download should begin automatically, saving the file in your default "download" location
  - Move these files to the folder in this directory: `data/viirscomps/` that was created when you initialized the `config.py` file in step 3. (Or you can set this to be your browser's download location.)
    - Note 1: This method uses the "stray-light corrected" composites. However there is no stray-light corrected version for the 2013 composite. 
    - Note 2: 2012 is only a partial year, so we'll ignore that file for now.
    - Note 3: when you first download a file you'll be prompted to create a user account. This is free, quick, and only needs to be done once.
  - Make sure these files are saved to the `data/viirscomps/` folder in this directory that was created when you initialized the `config.py` file in step 3.

These are large files and may take up to an hour to download them all, depending on internet bandwidth.

#### 4C: Uncompress all files
After the steps above, your `data/dmspcomps` and `data/viirscomps` folders will have all your files, but they're compressed.

To extract DMSP-OLS files:
1. For DMSP-OLS, open or extract each `tar` folder. (e.g. in MacOS, select all files, right-click, and select "open" -- for other OS use a utility of choice)
2. Your `data/dmspcomps` folder will now contain sub-folders for each satellite year, that contain more compressed files (e.g. `F101991.v4`)
3. To extract only the files we need and remove the rest, use the `unzipDMSP()` function in `harmonizer/downloader.py`.
  - In CLI from root: `$ python -m harmonizer.downloader -dmspunzip`

To extract VIIRS-DNB files:
1. After downloading, your `data/viirscomps` folder will contain several `.gz` files.
2. To extract, use the `unzipVIIRS()` function in `harmonizer/downloader.py`
  - In CLI from root: `$ python -m harmonizer.downloader -vunzip`

You should now have all the geoTIF files you need (and only what you need) in both `data/dmspcomps` and `data/viirscomps`

**Unless you delete these files, you dont need to download or uncompress them again. For any ROI you choose, these will serve as the source data.**

For more background information on both DMSP-OLS and VIIRS-DNB platforms, checkout the World Bank's "Open Nighttime Lights" tutorial:
https://worldbank.github.io/OpenNightLights/tutorials/mod1_2_introduction_to_nighttime_light_data.html

### Notes on future versions and data
Future releases will leverage the World Bank's "Light Every Night" (WB-LEN) archive, which contains nightly files and soon will contain monthly and annual composites as well:
https://registry.opendata.aws/wb-light-every-night/

Future releases will also seek to leverage the STAC/COG (https://stacindex.org/) convention (via the WB-LEN archive) and reduce the amount of data
required to be loaded locally.

This will remove the need to download full composites (i.e. no need for step 4 above), allowing the user to query only the ROI needed for a given program run, thus saving time and storage space.

Future versions of this package will probably include more advanced ways of creating an ROI on the fly.

## Using the program
### Get your Region Of Interest
This method requires an ROI to be saved locally as a shapefile. This can be a bounding box or a multipolygon (i.e. administrative boundary).

The `harmonizer/utils.py` script contains a method (`save_shp_from_url`) for saving a shapefile from the GADM website if you provide the url of a specific country shapefile.

The website http://geojson.io/ can be helpful for defining an ROI and saving locally as shapefile.

The only current requirement is that this geometry is formatted as a ESRI shapefile and that it's projected in the EPSG:4326 CRS in order to 
match the CRS the input nighttime lights files. Savy users can adjust the CRS as needed, but must remember to do so with all geo files involved (including the nighttime lights rasters) 
so that everything is projected in the same CRS.

However you get this shapefile, save it in the folder `roifiles` and set this location in your `config.py` file for the `roipath` variable.

**There are several country shapefiles already provided in the `roifiles` sub-directory if you wish to use them (uncomment the corresponding path in the `config.py` file).**

### Execute program
1. Make sure all data are downloaded and variables set, including the ROI shapefile for a particular run.
2. Execute the `main()` function in `harmonizer/main.py`. This takes 1 argument `trialname`: the name you'd like to call this run. e.g. "italy" if you're testing this on Italy.

To run this from CLI (in the root directory, i.e. location of this README) enter:`$ python -m harmonizer.main -n <yourtrialname>` (e.g. `$ python -m harmonizer.main italy`)

The end-to-end processing should take about 3 minutes to complete, depending on the size of your ROI and your hardware.

### Output
Running this process produces the following outputs:
1. A harmonized time series of every nighttime lights annual composite from 1992 to present:
  - These will be found in the "OUTPUT" directory (at the location you set in your `config.py` file) in side a sub-directory named after the "trialname" you set when executing the program.
  - These are single band geoTIFF files bounding the ROI you provided
    - Each pixel's value corresponds radiance as per a "Digital Number" that is the unit used in the DMSP-OLS data series
      (this model transforms the VIIRS-DNB radiance in nW/cm2/sr to a "Digital Number" unit like the DMSP-OLS)
  - DMSP-OLS composites have been adjusted (per the intercalibration process in `harmonizer/transformers/dmspcatlibrate.py`)
  - VIIRS-DNB composites have been preprocessed via the steps in `harmonizer/transormers/viirsprep.py` and the final model in `harmonizer/transformers/harmonize.py`
- NOTE: the default setting is to downsample VIIRS-DNB raster to be the same spatial resolution as DMSP-OLS (30 arc-seconds). However you can change the parameters (`DOWNSAMPLEVIIRS=False`) 
  in the `config.py` file which will not downsample VIIRS-DNB and leave them in the original 15 arc-second resolution. Please note that this may significantly affect analysis comparing data
  across both platforms (see the notes in `harmonizer/transformers/harmonize.py` for more detail.)
2. In the `results` directory (created at the location set in `config.py`) there will be several diagnostic plots describing the output, including comparisons 
of DMSP-OLS and the adjusted VIIRS-DNB data for your ROI during the year 2013, the training data year and the year when data from both platforms are available.
   
#### Diagnostics and metrics
The diagnostic plots and files are being developed but they include:
- `2013_comparison_raster.tif`: a geoTIF of your ROI that has 2 bands: 1st is the "harmonized" VIIRS-DNB data for 2013 and 2nd is the DMSP-OLS data for 2013
- `2013hist.png` a histogram of 2013 VIIRS-DNB and DMSP-OLS for your ROI (and results from a non-parametric test of distribution)
- `2013scatter.png` a scatterplot of 2013 VIIRS-DNB and DMSP-OLS (and Root Mean Squared Deviation (RMSD) and Spearman R correlation of the two datasets)
- `harmonized_ts_mean.png` a plot of the full time series showing mean radiance per pixel (default omits zero and near-zero light pixels)
- `harmonized_ts_median.png` a plot of the full time series showing median radiance per pixel (default omits zero and near-zero light pixels)
- `harmonized_ts_sum.png` a plot of the full time series showing sum of all pixels in ROI (default omits zero and near-zero light pixels)
- `final.mp4` a video of the full time series after harmonization
- `raw.mp4` a video of the full time series before harmonization
- `scatters_by_year/` a sub-directory containing scatter plots that compare harmonized and non-harmonized data for each year and states the Spearman R of the two. 
Since the DMSP-OLS had only minor "intercalibration" adjustments, the correlation is nearly 1.0; however, depending on the ROI the VIIRS-DNB correlations will be lower.

## Parameters
To run this process using default options you only need to follow the instructions above. These default parameters have been optimized based on testing and
evlatuion of several global test cases.

However, savvy users may decide to explore the parameters of various steps in the process (DMSP-OLS intercalibraiton, VIIRS-DNB preprocessing, and the DMSP-VIIRS
harmonize model that uses a Gradient Boosting Machine: XGBoost. Any and all parameters for these steps can be changed if you wish to customize the package further.)

## Methodology, technical specifications, and tutorial
More technical specification on the methodology and analysis will be presented in a forthcoming paper.

A basic tutorial on use is also being developed that will be hosted in the World Bank's "Open Nighttime Lights" repo and tutorial: https://worldbank.github.io/OpenNightLights/welcome.html 
