from pathlib import Path
import os


#########################
# ROI FILE
#########################

# The following country boundaries are included
# as test examples. To use them, uncomment the corresponding line below
# To provide your own shapefile, save it in the "roifiles/" folder
# and set the relative path to the roipath variable following these examples

# roipath = "roifiles/conus_shp/conus.shp"
# roipath = "roifiles/gadm36_DEU_shp/gadm36_DEU_0.shp"
# roipath = "roifiles/gadm36_ESP_shp/gadm36_ESP_0.shp"
# roipath = "roifiles/gadm36_FRA_shp/gadm36_FRA_0.shp"
# roipath = "roifiles/gadm36_ITA_shp/gadm36_ITA_0.shp"
# roipath = "roifiles/gadm36_JPN_shp/gadm36_JPN_0.shp"
# roipath = "roifiles/gadm36_MUS_shp/gadm36_MUS_0.shp"
# roipath = "roifiles/gadm36_NIC_shp/gadm36_NIC_0.shp"
# roipath = "roifiles/gadm36_PRI_shp/gadm36_PRI_0.shp"
# roipath = "roifiles/gadm36_SLV_shp/gadm36_SLV_0.shp"
roipath = "roifiles/gadm36_UGA_shp/gadm36_UGA_0.shp"

###################################
# OPTIONAL CHANGES
###################################

# set GDAL re-sampling algorithm (or leave as this default)
# options listed here: https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r
SAMPLEMETHOD = "average"

# set option for resampling (downsample VIIRS-DNB or leave as default True)
# more details described in the harmonizer/harmonize.py script
DOWNSAMPLEVIIRS = True

###################################
# YOU SHOULDNT NEED TO CHANGE THESE
###################################
ROOT = Path.cwd()
DMSP_URLS = Path(ROOT, "files", "noaa_dmsp_annual_urls.txt")
VIIRS_URLS = Path(ROOT, "files", "eog_viirs_annualv2_urls.txt")
ROIPATH = Path(ROOT, roipath)
DATA = Path(ROOT, "data")
DATA.mkdir(exist_ok=True)
DMSP_IN = Path(
    DATA, "dmspcomps"
)  # if you manually download DMSP files, save them here (e.g. <pathtothisrepo>/data/dmspcomps)
DMSP_IN.mkdir(exist_ok=True)
VIIRS_IN = Path(
    DATA, "viirscomps"
)  # if you manually download VIIRS files, save them here (e.g. <pathtothisrepo>/data/viirscomps)
VIIRS_IN.mkdir(exist_ok=True)
TMP = Path(DATA, "tmp")
TMP.mkdir(exist_ok=True)
DMSP_TMP = Path(TMP, "dmsp")
DMSP_TMP.mkdir(exist_ok=True)
VIIRS_TMP = Path(TMP, "viirs")
VIIRS_TMP.mkdir(exist_ok=True)
DMSP_CLIP = Path(TMP, "dmspclip")
DMSP_CLIP.mkdir(exist_ok=True)
VIIRS_CLIP = Path(TMP, "viirsclip")
VIIRS_CLIP.mkdir(exist_ok=True)
STAGE_TMP = Path(TMP, "staging")
STAGE_TMP.mkdir(exist_ok=True)
ARTIFACTS = Path(ROOT, "artifacts")
ARTIFACTS.mkdir(exist_ok=True)
OUTPUT = Path(ROOT, "output")
OUTPUT.mkdir(exist_ok=True)
RESULTS = Path(ROOT, "results")
RESULTS.mkdir(exist_ok=True)


# based on Li et al 2020
DMSP_PREFERRED_SATS = {
    "F10": ["1992", "1993", "1994"],
    "F12": ["1995", "1996"],
    "F14": ["1997", "1998", "1999", "2000", "2001", "2002", "2003"],
    "F16": ["2004", "2005", "2006", "2007", "2008", "2009"],
    "F18": ["2010", "2011", "2012", "2013"],
}
