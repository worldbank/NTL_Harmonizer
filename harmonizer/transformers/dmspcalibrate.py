import numpy as np
import rasterio, rasterio.mask
from pathlib import Path
from harmonizer.utils import clip_arr


PUBLISHED_COEFS = {
    "F14": np.asarray([-0.0781, 1.4588, -0.0073]),
    "F15": np.array([-0.4597, 1.714, -0.0114]),
    "F16A": {
        "F162004": np.array([0.1194, 1.2265, -0.0041]),
        "F162005": np.array([-0.3209, 1.4619, -0.0072]),
        "F162006": np.array([0.0877, 1.1616, -0.0021]),
        "F162007": np.array([0, 1, 0]),
        "F162008": np.array([0.1100, 1.0513, -0.001]),
        "F162009": np.array([0.6294, 1.1188, -0.0024]),
    },
    "F16B": np.array([-1.2802, 1.3313, -0.0055]),
    "F18": np.array([-0.0861, 0.821, 0.002]),
}


class DMSPstepwise:
    """
    Source: https://www.mdpi.com/2072-4292/9/6/637
    This method uses the co-efficients for each step
    derived by the reference source paper (above).

    But it's worth exploring a local-ly derived version of this
    with empirically derived co-efficients.

    Methodology:
    - Adjust F14 for underestimation (regression of overlapping F12, F14 years: 1997, 1998, 1999)
    - Period-based adjustment for F15
    - hybrid calibration for F16
    - adjustment of F18 2010 for overestimation
    """

    def __init__(self, dstdir):
        self.dstdir = dstdir
        # using published coefs
        self.f14coefs = PUBLISHED_COEFS["F14"]
        self.f15coefs = PUBLISHED_COEFS["F15"]
        self.f16Acoefs = PUBLISHED_COEFS["F16A"]
        self.f16Bcoefs = PUBLISHED_COEFS["F16B"]
        self.f18coefs = PUBLISHED_COEFS["F18"]

    def get_polyterms(self, X):
        X = X.flatten()
        X = np.stack([np.ones(X.shape, dtype=X.dtype), X, X ** 2]).T
        return X

    def get_F16coefs_from_path(self, srcpath):
        key = [k for k in self.f16Acoefs.keys() if k in str(srcpath)]
        assert len(key) == 1
        return self.f16Acoefs[key[0]]

    def process(self, X, coefs):
        shp = X.shape
        X = self.get_polyterms(X)
        X = np.dot(X, coefs.T).reshape(shp)
        X = clip_arr(X, floor=0.0, ceiling=63.0)
        return X

    def transform(self, srcpath):
        with rasterio.open(srcpath) as src:
            X = src.read(1)
            profile = src.profile

        if "F14" in str(srcpath):
            X = self.process(X, self.f14coefs)
        elif any(
            sat in str(srcpath)
            for sat in ["F152003", "F152004", "F152005", "F152006", "F152007"]
        ):
            X = self.process(X, self.f15coefs)
        elif "F16" in str(srcpath):
            coefs = self.get_F16coefs_from_path(srcpath)
            X = self.process(X, coefs)
            X = self.process(X, self.f16Bcoefs)
        elif "F182010" in str(srcpath):
            X = self.process(X, self.f18coefs)
        else:
            X = clip_arr(X, floor=0.0, ceiling=63.0)

        dstpath = Path(self.dstdir, srcpath.name)
        with rasterio.open(dstpath, "w", **profile) as dst:
            dst.write(X, 1)
