import requests, zipfile, io, re, os, rasterio, gzip, shutil, urllib.request, concurrent.futures
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial

#################
# RASTER OPS
#################


def crop_by_geom(srcpath, dstdir, geompath):
    dstpath = Path(dstdir, srcpath.name)
    os.system(
        f"gdalwarp -cutline {str(geompath)} -crop_to_cutline {str(srcpath)} {str(dstpath)}"
    )

def crop_by_geom_cwhere(srcpath, dstdir, geompath, query_val, query_attr):
    dstpath = Path(dstdir, srcpath.name)
    os.system(
        f'gdalwarp -cutline {str(geompath)} -cwhere "{query_attr}=\'{query_val}\'" -crop_to_cutline {str(srcpath)} {str(dstpath)}'
    )


def resample_raster(srcpath, refpath, dstdir, method):
    with rasterio.open(refpath) as src:
        width = src.width
        height = src.height
    dstpath = Path(dstdir, srcpath.name)
    os.system(
        f"gdalwarp -ts {width} {height} -r {method} {str(srcpath)} {str(dstpath)}"
    )


def compare_rasters(srcpath1, srcpath2, processfunc):
    with rasterio.open(srcpath1) as src:
        arr1 = src.read(1)

    with rasterio.open(srcpath2) as src:
        arr2 = src.read(1)

    return processfunc(arr1, arr2)


#################
# MATRIX OPS
#################

def clip_arr(arr, floor=None, ceiling=None):
    if floor is not None:
        arr[arr < floor] = floor
    if ceiling is not None:
        arr[arr > ceiling] = ceiling
    return arr


def gaussian_kernel(y, x, sigma):
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))


# def quartic_kernel(u):
#     return (15/16)*(1 - u**2)**2 if abs(u) <= 1 else 0

def get_kernel(pixelradius, sigma):
    y, x = np.ogrid[-pixelradius : pixelradius + 1, -pixelradius : pixelradius + 1]
    h = gaussian_kernel(y, x, sigma=sigma)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def sample_arr(arr, samplesize):
    if isinstance(samplesize, float):
        samplesize = int(samplesize * arr.shape[1])
    idx = np.random.choice(np.arange(arr.shape[1]), samplesize, replace=False)
    return arr[:, idx.tolist()]

# def fsigmoid(x, a, b, c, d):
#     return a + b* (1. / (1 + np.exp(-c*(x - d))))

def fsigmoid(x, L, x0, k, b):
    # adapted from: https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    return L / (1 + np.exp(-k * (x - x0))) + b

#################
# GENERAL UTILITIES
#################

def shp_from_url(url, dstdir):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=dstdir)


def save_shp_from_url(url, geodir):
    """
    creates a sub-dir in "geodir" based on the name of the url
    loads a shapefile form the url and saves the shapefile locally

    url: appropriately formatted url (GADM shapefile link)
    geodir: existing folder to save shapefile
    returns: filepath of the saved file
    """
    stem = Path(url).stem
    dstdir = Path(geodir, stem)
    dstdir.mkdir(exist_ok=True)
    dstpath = Path(dstdir, re.sub("_shp", "_0.shp", stem))
    shp_from_url(url, dstdir)
    return str(dstpath)


def clear_subdirs(subdirs):
    for subdir in subdirs:
        os.system(f"rm -rf {str(subdir)}")
        subdir.mkdir(exist_ok=True)


def filepathsearch(srcdir, keypattern, filepattern="*.tif", firstonly=True):
    fpaths = srcdir.glob(f"*{keypattern}{filepattern}")
    return next(fpaths) if firstonly else sorted(list(fpaths))


def download_from_url(url, dstdir):
    dstpath = Path(dstdir, Path(url).name)
    urllib.request.urlretrieve(url, dstpath)
    print(f"file saved at {dstpath}")


def batch_download(urls, dstdir, n_jobs):
    if n_jobs == 1:
        for url in tqdm(urls):
            download_from_url(url, dstdir)
    else:
        with concurrent.futures.ThreadPoolExecutor(n_jobs) as executor:
            func = partial(download_from_url, dstdir=dstdir)
            list(tqdm(executor.map(func, urls), total=len(urls)))


# def unpack_archive(srcpath, member, dstdir):
#     dstpath = Path(dstdir, srcpath.stem)
#     f = tarfile.TarFile(srcpath)
#     f.extract(member, )


# def unzip_tar(url, dstdir, member=None):
#     r = requests.get(url)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     if member is not None:
#         z.extract(member, dstdir)
#     else:
#         z.extractall(dstdir)
