from scipy.optimize import curve_fit
from harmonizer.utils import fsigmoid
import numpy as np

class CurveFit:
    def __init__(self, objfunc=fsigmoid):
        self.objfunc = objfunc

    def fit(self, X, y, epochs=None):
        p0 = [max(y), np.median(X), 1, min(y)]
        self.popt, _ = curve_fit(self.objfunc, X, y, p0=p0, method="dogbox")

    def predict(self, X):
        return self.objfunc(X, *self.popt)
