from harmonizer.utils import batch_download
from harmonizer.config import DMSP_URLS, DMSP_IN, VIIRS_IN
import argparse, gzip, shutil, os
from tqdm import tqdm
from pathlib import Path


def read_urls(srcpath):
    with open(srcpath, "r") as f:
        return f.read().split("\n")


def batch_download_DMSP(dmspurls=DMSP_URLS, dmspdir=DMSP_IN, n_jobs=16):
    dmspurls = read_urls(dmspurls)
    print("downloading DMSP-OLS composites...")
    batch_download(dmspurls, dmspdir, n_jobs)


def unzipDMSP(srcdir=DMSP_IN, pattern="**/*stable_lights.avg_vis.tif.gz"):
    # TODO: pool process
    srcpaths = srcdir.glob(pattern)
    for srcpath in tqdm(srcpaths):
        with gzip.open(srcpath, "rb") as f_in:
            with open(Path(srcpath.parent.parent, srcpath.stem), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    dpaths = srcdir.glob("*.v4")
    for dpath in dpaths:
        os.system(f"rm -rf {dpath}")


def unzipVIIRS(srcdir=VIIRS_IN, pattern="*.gz"):
    # TODO: pool process
    srcpaths = srcdir.glob(pattern)
    for srcpath in tqdm(srcpaths):
        with gzip.open(srcpath, "rb") as f_in:
            with open(Path(srcpath.parent, srcpath.stem), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.system(f"rm {srcpath}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dmsp", "--dmsp", action="store_true", help="download DMSP files"
    )
    parser.add_argument(
        "-dmspunzip", "--dmspunzip", action="store_true", help="unzip DMSP files"
    )
    parser.add_argument(
        "-vunzip", "--vunzip", action="store_true", help="unzip VIIRS files"
    )
    args = parser.parse_args()
    return args.dmsp, args.dmspunzip, args.vunzip


if __name__ == "__main__":
    dmsp, dmspunzip, vunzip = get_args()
    if dmsp:
        batch_download_DMSP()
    if dmspunzip:
        unzipDMSP()
    if vunzip:
        unzipVIIRS()
