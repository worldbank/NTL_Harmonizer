import rasterio, rasterio.mask, dill
from pathlib import Path
from harmonizer.utils import clip_arr, filepathsearch, resample_raster
import numpy as np


class Harmonizer:
    """
    The current implementation assumes annual composites.
    It would need some adjustments to allow monthly composites.

    This model provides 2 choices regarding VIIRS-DNB spatial resolution.

    Option 1 (default): VIIRS-DNB data are down-sampled to 30 arc seconds in order to correspond to the DMSP-OLS
    data on a pixel-by-pixel bases so the model can be fit. All VIIRS-DNB data must then be down-sampled to 30 arc secs
    so that this models weights can be applied. This produces a final time series output where all files
    from both platforms are at 30 arc sec resolution.

    Option 2: DMSP-OLS training data (e.g. 2013 annual composite) are up-sampled to 15 arc sec resolution
    to correspond to VIIRS-DNB on a pixel-by-pixel bases so the model can be fit. The model weights can then be applied
    to all VIIRS-DNB data at its original 15 arc sec resolution. This preserves the VIIRS-DNB original pixel size;
    however, the VIIRS-DNB signal is still degraded to match DMSP-OLS (thanks to the model adjustments).

    This produces a final time series where the DMSP-OLS are maintained at original 30 arc sec resolution and the VIIRS-DNB
    rasters are maintained at their original 15 arc sec resolution. Since the platforms have different pixel resolutions,
    comparison across both platforms requires standardization (i.e. avg per pixel radiance, etc).
    Geometric aggregations, such as Sum Of Lights, would produce incomparable measures unless the data were first standardized per pixel.

    A third option would be to up-sample all DMSP-OLS data (not just that used for modeling training) so that the entire
    time series has 15 arc sec resolution. This is less desirable as it implies more precision in the DMSP-OLS data
    than actually exists -- in contrast to downsampling VIIRS-DNB which reduces existing information. This is not
    presented as an option currently.

    """

    def __init__(
        self,
        dmspdir,
        viirsdir,
        stagedir,
        output,
        downsampleVIIRS,
        samplemethod,
        polyX,
        idX,
        epochs,
        est,
        opath,
    ):
        self.dmsp_train = filepathsearch(dmspdir, "2013", "*.tif")
        self.viirs_train = filepathsearch(viirsdir, "2013", "*.tif")
        self.stagedir = stagedir
        self.output = output
        self.downsampleVIIRS = downsampleVIIRS
        self.samplemethod = samplemethod
        self.polyX = polyX
        self.idX = idX
        self.epochs = epochs
        self.est = est
        self.opath = opath

    def load_arr(self, srcpath):
        with rasterio.open(srcpath) as src:
            return src.read(1)

    def Xy_transform(self, X, y=None):
        X = np.expand_dims(X, axis=0)
        if self.polyX:
            X = np.concatenate([np.ones(X.shape), X, X ** 2], axis=0)
        if self.idX:
            idgrid = np.meshgrid(range(0, X.shape[-1]), range(0, X.shape[-2]))
            xgrid = np.expand_dims(idgrid[0], axis=0)
            ygrid = np.expand_dims(idgrid[1], axis=0)
            X = np.concatenate([X, xgrid, ygrid], axis=0)
        X = X.reshape(X.shape[0], -1).T
        if y is not None:
            y = y.flatten()
        return X, y

    def train_prep(self):
        if self.downsampleVIIRS:
            resample_raster(
                srcpath=self.viirs_train,
                refpath=self.dmsp_train,
                dstdir=self.stagedir,
                method=self.samplemethod,
            )
            self.viirs_train = Path(self.stagedir, self.viirs_train.name)
        else:  # upsample DMSP...
            resample_raster(
                srcpath=self.dmsp_train,
                refpath=self.viirs_train,
                dstdir=self.stagedir,
                method=self.samplemethod,
            )
            self.dmsp_train = Path(self.stagedir, self.dmsp_train.name)
        X = self.load_arr(self.viirs_train)
        y = self.load_arr(self.dmsp_train)
        X, y = self.Xy_transform(X, y)
        return X, y

    def fit(self):
        X, y = self.train_prep()
        self.est.fit(X, y, self.epochs)

    def process(self, X):
        shp = X.shape
        dtype = X.dtype
        X, _ = self.Xy_transform(X)
        X = self.est.predict(X)
        X = clip_arr(X, floor=0, ceiling=63.0)
        return X.reshape(shp).astype(dtype)

    def transform(self, srcpath):
        if self.downsampleVIIRS:
            resample_raster(
                srcpath,
                refpath=self.dmsp_train,
                dstdir=self.output,
                method=self.samplemethod,
            )
            srcpath = Path(self.output, srcpath.name)
        with rasterio.open(srcpath) as src:
            X = src.read(1)
            profile = src.profile
        X = self.process(X)
        dstpath = Path(self.output, srcpath.name)
        with rasterio.open(dstpath, "w", **profile) as dst:
            dst.write(X, 1)


def save_obj(obj, opath):
    with open(opath, "wb") as dst:
        dill.dump(obj, dst)
    print(f"model artifact saved at {str(opath)}")


def load_obj(srcpath):
    with open(srcpath, "rb") as src:
        return dill.load(src)
