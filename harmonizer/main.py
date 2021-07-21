from harmonizer.config import (
    ROOT,
    VIIRS_IN,
    VIIRS_TMP,
    DMSP_IN,
    DMSP_TMP,
    DMSP_CLIP,
    VIIRS_CLIP,
    STAGE_TMP,
    RESULTS,
    DMSP_PREFERRED_SATS,
    SAMPLEMETHOD,
    DOWNSAMPLEVIIRS,
    ROIPATH,
    ARTIFACTS,
    OUTPUT,
)
import time, argparse
from pathlib import Path
from harmonizer.utils import clear_subdirs, crop_by_geom
from harmonizer.transformers.dmspcalibrate import DMSPstepwise
from harmonizer.transformers.viirsprep import VIIRSprep
from harmonizer.transformers.harmonize import Harmonizer, save_obj
from harmonizer.transformers.gbm import XGB
from harmonizer.transformers.curve import CurveFit
from harmonizer.diagnostics import main as diagnosticsmain
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from tqdm import tqdm


def crop_batch(srcdir, dstdir, geompath, n_jobs):
    srcpaths = srcdir.glob("*.tif")
    if n_jobs > 1:
        func = partial(crop_by_geom, dstdir=dstdir, geompath=geompath)
        thread_pool = ThreadPool(n_jobs)
        _ = tqdm(thread_pool.imap(func, srcpaths))
    else:
        for srcpath in tqdm(srcpaths):
            crop_by_geom(srcpath, dstdir=dstdir, geompath=geompath)


def dmsp_batch(srcdir, dstdir, selectDMSP):
    sats = sorted([k + yr for k, v in selectDMSP.items() for yr in v])
    srcpaths = [next(srcdir.glob(f"*{sat}*.tif")) for sat in sats]
    calibrator = DMSPstepwise(dstdir=dstdir)
    for srcpath in tqdm(srcpaths):
        calibrator.transform(srcpath)


def viirs_batch(srcdir, dstdir):
    srcpaths = srcdir.glob("*.tif")
    prepper = VIIRSprep(
        pixelradius=5,
        sigma=2.0, # 0.25
        damperthresh=1.0,
        usedask=True,
        chunks="auto",
        dstdir=dstdir,
    )
    for srcpath in tqdm(srcpaths):
        prepper.transform(srcpath)


def harmonize_batch(dmspdir, viirsdir, stage_dir, output, artifactpath):
    srcpaths = viirsdir.glob("*.tif")
    finalharmonizer = Harmonizer(
        dmspdir=dmspdir,
        viirsdir=viirsdir,
        stagedir=stage_dir,
        output=output,
        downsampleVIIRS=DOWNSAMPLEVIIRS,
        samplemethod=SAMPLEMETHOD,
        polyX=False,
        idX=False,
        epochs=100,
        est=CurveFit(),
        opath=artifactpath,
    )
    print("training model on ROI data...")
    finalharmonizer.fit()
    for srcpath in tqdm(srcpaths):
        finalharmonizer.transform(srcpath)
    save_obj(finalharmonizer, finalharmonizer.opath)


def main(trialname, crop, roipath=ROIPATH):
    trialout = Path(OUTPUT, trialname)
    trialout.mkdir(exist_ok=True)
    trialresults = Path(RESULTS, trialname)
    trialresults.mkdir(exist_ok=True)
    clear_subdirs(
        [DMSP_TMP, VIIRS_TMP, STAGE_TMP, trialout, trialresults]
    )  # this will overwrite previously processed files

    # clip nighttime files to ROI
    t0 = time.time()
    if crop:
        clear_subdirs([DMSP_CLIP, VIIRS_CLIP])
        crop_batch(srcdir=DMSP_IN, dstdir=DMSP_CLIP, geompath=roipath, n_jobs=1)
        crop_batch(srcdir=VIIRS_IN, dstdir=VIIRS_CLIP, geompath=roipath, n_jobs=1)
        print(f"time to crop: {time.time() - t0:.4f}s")

    # calibrate DMSP-OLS using the stepwise method
    t1 = time.time()
    dmsp_batch(srcdir=DMSP_CLIP, dstdir=trialout, selectDMSP=DMSP_PREFERRED_SATS)
    print(f"time to calibrate dmsp: {time.time() - t1:.4f}s")

    # preprocess VIIRS-DNB
    t2 = time.time()
    viirs_batch(srcdir=VIIRS_CLIP, dstdir=VIIRS_TMP)
    print(f"time to preprocess viirs: {time.time() - t2:.4f}s")

    # harmonize
    t3 = time.time()
    harmonize_batch(
        dmspdir=trialout,
        viirsdir=VIIRS_TMP,
        stage_dir=STAGE_TMP,
        output=trialout,
        artifactpath=Path(ARTIFACTS, "harmonizer.dill"),
    )
    print(f"time to harmonize: {time.time() - t3:.4f}s")

    # diagnostics
    t4 = time.time()
    diagnosticsmain(
        outputdir=trialout,
        resultsdir=trialresults,
        dmsp_clip=DMSP_CLIP,
        viirs_clip=VIIRS_CLIP,
        viirs_tmp=VIIRS_TMP,
        selectDMSP=DMSP_PREFERRED_SATS,
        lowerthresh=3,
    )
    print(f"time to plot results: {time.time() - t4:.4f}s")
    print("DONE!")
    print(f"final time series located at: {trialout}")
    print(f"diagnostic plots and metrics at: {trialresults}")
    print(f"total runtime: {time.time() - t0:.4f}s")
    return

def process_all_test_samples():
    # this runs the harmonizer and creates results plots
    # on each of the shapefiles in the `roifiles/` directory.
    t0 = time.time()
    rois = Path(ROOT,"roifiles").glob("**/*.shp")
    for roi in tqdm(rois):
        print("*"*25)
        print(f"processing {roi.stem}...")
        print("*" * 25)
        roipath = roi
        main(trialname=roipath.stem, crop=True, roipath=roipath)
    print("DONE!")
    print(f"total runtime: {time.time() - t0:.4f}s")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="test", help="trial name")
    parser.add_argument("-nc", "--nocrop", action="store_false", help="skips cropping")
    parser.add_argument("-a", "--alltest", action="store_true", help="runs all test cases in roifiles")
    args = parser.parse_args()
    return args.name, args.nocrop, args.alltest


if __name__ == "__main__":
    trialname, crop, alltest = get_args()
    if alltest:
        process_all_test_samples()
    else:
        main(trialname, crop)
