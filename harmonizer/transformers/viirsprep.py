import numpy as np
from scipy.ndimage import convolve
from dask_image.ndfilters import convolve as dconvolve
import dask.array as da
import rasterio
from pathlib import Path
from harmonizer.utils import get_kernel


class VIIRSprep:
    def __init__(self, pixelradius, sigma, damperthresh, usedask, chunks, dstdir):
        self.pixelradius = pixelradius
        self.sigma = sigma
        self.damperthresh = damperthresh
        self.usedask = usedask
        self.chunks = chunks
        self.dstdir = dstdir
        self.kernel = get_kernel(self.pixelradius, self.sigma)

    def convolve_arr(self, arr, mode="constant", cval=0.0, **kwargs):
        return (
            dconvolve(arr, weights=self.kernel, mode=mode, cval=cval, **kwargs)
            if self.usedask
            else convolve(arr, weights=self.kernel, mode=mode, cval=cval, **kwargs)
        )

    @staticmethod
    def damper(X, thresh, targetval):
        X[X <= thresh] = targetval
        return X

    def logtransform(self, X):
        return da.log1p(X) if self.usedask else np.log1p(X)

    def process(self, X):
        if self.usedask:
            X = da.from_array(X, chunks=self.chunks)
        X = self.damper(X, thresh=self.damperthresh, targetval=0)
        X = self.convolve_arr(X)
        X = self.logtransform(X)
        return X.compute() if self.usedask else X

    def transform(self, srcpath):
        with rasterio.open(srcpath) as src:
            X = src.read(1)
            profile = src.profile
        X = self.process(X)
        dstpath = Path(self.dstdir, srcpath.name)
        with rasterio.open(dstpath, "w", **profile) as dst:
            dst.write(X, 1)
