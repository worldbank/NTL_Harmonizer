from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class CurveFit:
    def __init__(self, degree=3):
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=degree)

    def fit(self, X, y, epochs=None):

        X = self.poly_features.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.poly_features.fit_transform(X)
        return self.model.predict(X)

class NoFit:
    def fit(self, X, y, epochs=None):
        return

    def predict(self, X):
        return X