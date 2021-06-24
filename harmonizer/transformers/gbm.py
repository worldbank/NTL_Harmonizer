import xgboost as xgb


XGB_PARAMS = {
    "subsample": 0.5,
    "n_jobs": 16,
    "random_state": 32,
    # "gpu_id":0,
    "max_depth": 6,
    "tree_method": "hist",
    "verbosity": 1,
}

XGB_CV_PARAMS = {"num_boost_round": 10, "nfold": 3, "stratified": False}


class XGB:
    def __init__(self, params=XGB_PARAMS, cvparams=XGB_CV_PARAMS, bst=None):
        self.params = params
        self.cvparams = cvparams
        self.bst = bst

    def fit(self, X, y, num_boost_rounds):
        dtrain = xgb.DMatrix(X, y)
        output = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_rounds,
            xgb_model=self.bst,
            evals=[(dtrain, "train")],
        )
        self.bst = output

    def cv(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        output = xgb.cv(params=self.params, dtrain=dtrain, **self.cvparams)
        self.cvoutput = output

    def predict(self, X):
        X = xgb.DMatrix(X)
        return self.bst.predict(X)

    def save_boost(self, opath):
        self.bst.save_model(opath)
