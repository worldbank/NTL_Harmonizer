import pandas as pd
import numpy as np
from pathlib import Path
import os, h5py
from harmonizer.utils import filepathsearch
from harmonizer.diagnostics import get_yr_from_path, get_series
from harmonizer.config import DMSP_PREFERRED_SATS

GDPFPATH= Path(os.environ['NLT'], "NTL_Harmonizer","files","nama_10r_3gdp.csv")

def convert_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def load_prep(srcpath=GDPFPATH):
    df = pd.read_csv(srcpath).drop(["UNIT", "Flag and Footnotes"], axis=1)
    idx = [("DE" in i) | ("ES" in i) | ("IT" in i) | ("FR" in i) for i in df.GEO]
    df = df.loc[idx, :]
    df['country'] = df["GEO"].str[:2]
    df['admin1'] = df['GEO'].str[2:3]
    df['admin2'] = df['GEO'].str[3:4]
    df['admin3'] = df['GEO'].str[4:]
    df['level'] = (df.loc[:, ["admin1", "admin2","admin3"]] != "").sum(axis=1)
    return df

def extract_time_series(dmsp_in,
                        viirs_in,
                        outputdir,
                        thresh,
                        n_jobs,
                        selectDMSP=DMSP_PREFERRED_SATS):
    sats = sorted([k + yr for k, v in selectDMSP.items() for yr in v])
    dmspraw = [next(dmsp_in.glob(f"*{sat}*.tif")) for sat in sats]
    viirsraw = list(viirs_in.glob("*.tif"))
    dmspout = filepathsearch(outputdir, "F", "*.tif", firstonly=False)
    dmspyrs = [get_yr_from_path(i) for i in dmspout]
    viirsout = filepathsearch(outputdir, "VNL", "*.tif", firstonly=False)
    viirsyrs = [get_yr_from_path(i) for i in viirsout]
    out = {}
    out["DMSP_unadj_mean"] = get_series(dmspraw, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_unadj_mean'] = get_series(viirsraw, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    out["DMSP_adj_mean"] = get_series(dmspout, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_adj_mean'] = get_series(viirsout, np.nanmean, thresh=thresh, n_jobs=n_jobs)

    out["DMSP_unadj_md"] = get_series(dmspraw, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_unadj_md'] = get_series(viirsraw, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    out["DMSP_adj_md"] = get_series(dmspout, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_adj_md'] = get_series(viirsout, np.nanmedian, thresh=thresh, n_jobs=n_jobs)

    out["DMSP_unadj_sol"] = get_series(dmspraw, np.nansum, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_unadj_sol'] = get_series(viirsraw, np.nansum, thresh=thresh, n_jobs=n_jobs)
    out["DMSP_adj_sol"] = get_series(dmspout, np.nansum, thresh=thresh, n_jobs=n_jobs)
    out['VIIRS_adj_sol'] = get_series(viirsout, np.nansum, thresh=thresh, n_jobs=n_jobs)

    out['DMSP_yrs'] = dmspyrs
    out["VIIRS_yrs"] = viirsyrs
    return out
