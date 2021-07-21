from pathlib import Path
import rasterio, os, rasterio.mask, imageio
from datetime import datetime
from scipy.stats import spearmanr, anderson_ksamp
import numpy as np
from tqdm import tqdm
from harmonizer.utils import sample_arr, filepathsearch, resample_raster
from harmonizer.plots import raster_scatter, raster_hist, plot_timeseries
from harmonizer.config import SAMPLEMETHOD, DOWNSAMPLEVIIRS
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

# 2013 metrics
def calc_rmsd(yhat, y):
    return np.sqrt(np.mean((y - yhat) ** 2))


def calc_r2(yhat, y):
    return spearmanr(yhat, y)


def calc_AD(yhat, y, binfirst):
    if binfirst:
        yhat = calc_bins(yhat)
        y = calc_bins(y)
    return anderson_ksamp((yhat, y))


def calc_bins(x, nbins=63):
    bins, edges = np.histogram(x, bins=nbins)
    return bins


def calc_normdiff(yhat, y):
    return (yhat - y) / (yhat + y + 1e-9)


def transform_array(arr1, arr2, thresh, samplesize=1000000):
    arr = np.stack((arr1, arr2))
    arr = arr.reshape(2, -1)
    if thresh is not None:
        idx = (arr[0, :] >= thresh) & (arr[1, :] >= thresh)
        arr = arr[:, idx]
    if arr.shape[1] > samplesize:
        arr = sample_arr(arr, samplesize)
    return arr


def training_diagnostics(srcdir, resultsdir, thresh=1):
    dpath = filepathsearch(srcdir, "F182013", "*.tif")
    vpath = filepathsearch(srcdir, "VNL*2013", "*.tif")
    with rasterio.open(dpath) as src:
        darr = src.read(1)
    with rasterio.open(vpath) as src:
        varr = src.read(1)
        vprofile = src.profile

    arr = transform_array(varr, darr, thresh)
    x = arr[0, :]
    y = arr[1, :]
    rmsd = calc_rmsd(x, y)
    r2, p = calc_r2(x, y)
    stat, cv, sig = calc_AD(x, y, binfirst=True)
    if sig > 0.0025:
        adresults = "AD test failed to reject null at 2.5% alpha.\nIndicates same population. (expected)."
    else:
        adresults = "AD test rejected null at 2.5% alpha.\nIndicates different population. (unexpected)."
    adlabel = f"AD stat: {stat:.4f} (est. p={sig})\n{adresults}"
    raster_scatter(
        x=x,
        y=y,
        xlabel="VIIRS-DNB (harmonized Digital Number)",
        ylabel="DMSP-OLS (DN)",
        title='2013 VIIRS-DNB (harmonized) vs DMSP-OLS ("lit pixels") for ROI',
        opath=Path(resultsdir, "2013scatter.png"),
        alpha=0.01,
        label=f"RMSD: {rmsd:.4f}\nSpearman R: {r2:.4f} (p={p})",
    )
    raster_hist(
        x=x,
        y=y,
        bins=63,
        label1="VIIRS-DNB (harmonized)",
        label2="DMSP-OLS",
        xlabel="Per pixel radiance (Digital number)",
        ylabel="Frequency",
        title='Distribution of 2013 VIIRS-DNB (harmonized) and DMSP-OLS per pixel radiance ("lit pixels") for ROI',
        opath=Path(resultsdir, "2013hist.png"),
        text=adlabel,
    )
    vprofile.update(count=2)
    with rasterio.open(
        Path(resultsdir, "2013_comparison_raster.tif"), "w", **vprofile
    ) as dst:
        for i, arr in enumerate([varr, darr]):
            dst.write(arr.astype(vprofile["dtype"]), i + 1)
        dst.descriptions = ("VIIRS-DNB (harmonized)", "DMSP-OLS")


# time series plots
def load_arr(srcpath):
    with rasterio.open(srcpath) as src:
        return src.read(1)


def reduce_array(arr, reducefunc, thresh):
    if thresh is not None:
        arr[arr < thresh] = np.nan
    return reducefunc(arr)

def extract_and_reduce(srcpath, reducfunc, thresh):
    X = load_arr(srcpath).astype(np.float32)
    return reduce_array(X, reducfunc, thresh)

def get_series(srcpaths, reducfunc, thresh, n_jobs=1):
    if n_jobs > 1:
        func = partial(extract_and_reduce,
                       reducfunc=reducfunc,
                       thresh=thresh)
        thread_pool = ThreadPool(n_jobs)
        series = list(thread_pool.imap(func, srcpaths))
    else:
        series = [extract_and_reduce(srcpath, reducfunc, thresh) for srcpath in tqdm(srcpaths)]
    return series

def get_yr_from_path(srcpath):
    for yr in range(1992, int(datetime.now().year) + 1):
        if str(yr) in str(srcpath):
            return yr

def plot_final_time_series(outputdir, resultsdir, thresh=None):
    dmsp = filepathsearch(outputdir, "F", "*.tif", firstonly=False)
    dmspyrs = [get_yr_from_path(i) for i in dmsp]
    viirs = filepathsearch(outputdir, "VNL", "*.tif", firstonly=False)
    viirsyrs = [get_yr_from_path(i) for i in viirs]
    dmsp_avg = get_series(dmsp, np.nanmean, thresh=thresh)
    viirs_avg = get_series(viirs, np.nanmean, thresh=thresh)
    plot_timeseries(
        seqs=[dmsp_avg, viirs_avg],
        yrs=[dmspyrs, viirsyrs],
        labels=["DMSP-OLS", "VIIRS-OLS (harmonized)"],
        opath=Path(resultsdir, "harmonized_ts_mean.png"),
        xlabel="Year",
        ylabel="mean radiance per pixel (DN)",
        title="Harmonized time series 1992-present for ROI (mean)",
    )

    dmsp_md = get_series(dmsp, np.nanmedian, thresh=thresh)
    viirs_md = get_series(viirs, np.nanmedian, thresh=thresh)
    plot_timeseries(
        seqs=[dmsp_md, viirs_md],
        yrs=[dmspyrs, viirsyrs],
        labels=["DMSP-OLS", "VIIRS-OLS (harmonized)"],
        opath=Path(resultsdir, "harmonized_ts_median.png"),
        xlabel="Year",
        ylabel="median radiance per pixel (DN)",
        title="Harmonized time series 1992-present for ROI (median)",
    )
    dmsp_sol = get_series(dmsp, np.nansum, thresh=thresh)
    viirs_sol = get_series(viirs, np.nansum, thresh=thresh)
    plot_timeseries(
        seqs=[dmsp_sol, viirs_sol],
        yrs=[dmspyrs, viirsyrs],
        labels=["DMSP-OLS", "VIIRS-OLS (harmonized)"],
        opath=Path(resultsdir, "harmonized_ts_sum.png"),
        xlabel="Year",
        ylabel="Sum of Lights (DN)",
        title="Harmonized time series 1992-present for ROI (SOL)",
    )


def year_scatter(
    srcpath,
    dmsp_clip,
    viirs_clip,
    viirs_tmp,
    dstdir,
    thresh,
    method=SAMPLEMETHOD,
    downsampleVIIRS=DOWNSAMPLEVIIRS,
):
    if Path(dmsp_clip, srcpath.name).exists():
        rawpath = Path(dmsp_clip, srcpath.name)
    else:
        rawpath = Path(viirs_clip, srcpath.name)
        if downsampleVIIRS:
            tmppath = Path(viirs_tmp, srcpath.name)
            if tmppath.exists():
                os.system(f"rm {str(tmppath)}")
            resample_raster(rawpath, srcpath, viirs_tmp, method)
            rawpath = Path(viirs_tmp, srcpath.name)
    with rasterio.open(rawpath) as src:
        raw = src.read(1)
    with rasterio.open(srcpath) as src:
        transformed = src.read(1)
    arr = transform_array(raw, transformed, thresh)
    x = arr[0, :]
    y = arr[1, :]
    r2, p = calc_r2(x, y)
    raster_scatter(
        x=x,
        y=y,
        xlabel="Input composite",
        ylabel="Processed image",
        title="Processed data vs input data for ROI",
        opath=Path(dstdir, f"{srcpath.name}_scatter.png"),
        alpha=0.5,
        label=f"Spearman R: {r2:.4f} (p={p})",
    )


def create_year_scatters(
    outputdir, dmsp_clip, viirs_clip, viirs_tmp, resultsdir, thresh
):
    dstdir = Path(resultsdir, "scatters_by_year")
    dstdir.mkdir(exist_ok=True)
    srcpaths = outputdir.glob("*.tif")
    for srcpath in tqdm(srcpaths):
        year_scatter(srcpath, dmsp_clip, viirs_clip, viirs_tmp, dstdir, thresh)


def make_movie(srcpaths, dstpath, fps):
    series = []
    for srcpath in tqdm(srcpaths):
        series.append(load_arr(srcpath).astype(np.float32))
    imageio.mimwrite(dstpath, series, fps=fps)


def create_all_movies(dmsp_clip, viirs_tmp, output, resultsdir, selectDMSP, fps=3):
    final = sorted(list(output.glob("F*.tif")))
    final += sorted(list(output.glob("*VNL*.tif")))
    sats = sorted([k + yr for k, v in selectDMSP.items() for yr in v])
    raw = [next(dmsp_clip.glob(f"*{sat}*.tif")) for sat in sats]
    raw += sorted(list(viirs_tmp.glob("*.tif")))
    make_movie(final, Path(resultsdir, "final.mp4"), fps)
    make_movie(raw, Path(resultsdir, "raw.mp4"), fps)


def main(outputdir, resultsdir, dmsp_clip, viirs_clip, viirs_tmp, lowerthresh, selectDMSP):
    training_diagnostics(srcdir=outputdir, resultsdir=resultsdir, thresh=lowerthresh)
    plot_final_time_series(
        outputdir=outputdir, resultsdir=resultsdir, thresh=lowerthresh
    )
    create_year_scatters(
        outputdir=outputdir,
        dmsp_clip=dmsp_clip,
        viirs_clip=viirs_clip,
        viirs_tmp=viirs_tmp,
        resultsdir=resultsdir,
        thresh=lowerthresh,
    )
    create_all_movies(
        dmsp_clip=dmsp_clip,
        viirs_tmp=viirs_tmp,
        output=outputdir,
        resultsdir=resultsdir,
        selectDMSP=selectDMSP,
        fps=3,
    )
