import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import os, h5py
from harmonizer.utils import filepathsearch
from harmonizer.diagnostics import get_yr_from_path, get_series
from harmonizer.config import DMSP_PREFERRED_SATS
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import spearmanr

def convert_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def load_gdp(srcpath):
    df = pd.read_csv(srcpath).drop(["UNIT", "Flag and Footnotes"], axis=1)
    idx = [("DE" in i) | ("ES" in i) | ("IT" in i) | ("FR" in i) for i in df.GEO]
    df = df.loc[idx, :]
    df['country'] = df["GEO"].str[:2]
    df['admin1'] = df['GEO'].str[2:3]
    df['admin2'] = df['GEO'].str[3:4]
    df['admin3'] = df['GEO'].str[4:]
    df['level'] = (df.loc[:, ["admin1", "admin2","admin3"]] != "").sum(axis=1)
    df['GDP'] = df['Value'].str.replace(',', '').str.replace(':','')
    df["GDP"] = df['GDP'].apply(lambda x: float(x) if x != '' else np.nan)
    return df

# def load_geoms(geompath=GEOMPATH, geomid=GEOMID):
#     gdf = gpd.read_file(geompath)
#     gdf["id"] = gdf["NUTS_BN_ID"]
#     ids = pd.read_csv(geomid)
#     ids["id"] = ids["NUTS_BN_CODE"]
#     return gdf.loc[:,['id','geometry']].merge(ids, on='id', how='left')

def select_geoms(gdp, gdf):
    idx = gdp["GEO"].unique()
    return gdf.set_index("NUTS_ID").loc[idx, ["NAME_LATN","geometry"]]

def get_country_data(nuts_code, gdp, gdf, ntl, level=0):
    gdp = gdp.loc[(gdp["country"] == nuts_code) & (gdp['level']==level), ["TIME","GEO","GDP"]]
    geoms = select_geoms(gdp, gdf)
    gdp.set_index(["GEO", "TIME"], inplace=True)
    return gdp.join(ntl), geoms

def get_ts(df, value_col):
    return df.pivot(index='TIME', columns=["GEO"], values=[value_col])

def extract_time_series(nuts_code,
                        dmsp_in,
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

    dseries = get_series(dmspraw, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsraw, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    yrs = dmspyrs + viirsyrs
    out = pd.DataFrame([yrs, dseries + vseries], index=["TIME","unadj_mu"]).T

    dseries = get_series(dmspout, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsout, np.nanmean, thresh=thresh, n_jobs=n_jobs)
    out['adj_mu'] = dseries + vseries

    dseries = get_series(dmspraw, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsraw, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    out['unadj_md'] = dseries + vseries

    dseries = get_series(dmspout, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsout, np.nanmedian, thresh=thresh, n_jobs=n_jobs)
    out['adj_md'] = dseries + vseries

    dseries = get_series(dmspraw, np.nansum, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsraw, np.nansum, thresh=thresh, n_jobs=n_jobs)
    out['unadj_sol'] = dseries + vseries

    dseries = get_series(dmspout, np.nansum, thresh=thresh, n_jobs=n_jobs)
    vseries = get_series(viirsout, np.nansum, thresh=thresh, n_jobs=n_jobs)
    out['adj_sol'] = dseries + vseries
    out['GEO'] = nuts_code
    out['TIME'] = out['TIME'].astype(int)
    return out.set_index(["GEO","TIME"])

def compare_ts_plot(df, gdpcol, ntl_un, ntladj, label1, label2, ylabel, title):
    fig, ax = plt.subplots(1, figsize=(15, 7))
    X = [datetime.strptime(str(i), "%Y").year for i in df.index.get_level_values(1)]
    ax = sns.lineplot(x=X, y=df[gdpcol], color='steelblue', lw=3, label=gdpcol)
    ax2 = ax.twinx()
    sns.lineplot(x=X, y=df[ntl_un], ax=ax2,
                 color='indianred', ls='--', label=label1)
    sns.lineplot(x=X, y=df[ntladj], ax=ax2, lw=3,
                 color='indianred', label=label2)
    ax.legend(loc='lower left',fontsize=12)
    ax2.legend(loc='lower right', fontsize=12)
    ax.set_ylim([0, 3500000])
    ax2.set_ylim([0, 17000000])
    ax.set_xlabel("Year", fontsize=20)
    ax.set_ylabel(gdpcol + " (current)", fontsize=20)
    ax2.set_ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20);

def compare_scatter(df, gdpcol, ntl_un, ntladj, title):
    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    for i, collab in enumerate(zip([ntl_un, ntladj],["NTL (unadj.)", "NTL (adj.)"])):
        r, p = spearmanr(df[gdpcol].dropna(), df[collab[0]].dropna())
        sns.regplot(x=collab[0], y=gdpcol, data=df, ax=ax[i], label=f"Spearman R: {r:.4f}, (p={p:.4f})")
        ax[i].set_xlabel(collab[1], fontsize=20)
        ax[i].set_ylabel("GDP (current)", fontsize=20)
        ax[i].legend(fontsize=20)
    plt.suptitle(title, fontsize=20)


# def concat_ntl_ts(ntldict, dmspkey, viirskey, dyrs="DMSP_yrs", vyrs="VIIRS_yrs"):
#
# def append_ntl(gdp, ntldict):
#     unadjmn =