from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import numpy as np
from .chain_ladder import ChainLadderRegressor


try:
    import databricks.koalas as kl
except ImportError:
    import pandas as kl

    pass


class PPCIRegressor(ChainLadderRegressor):
    """ Estimate ultimates based on PPCI """

    def __init__(self):
        """ Init """
        # self.input_X_datasets = ["origin", "claim_paid"]
        # self.input_y_datasets = None
        # self.origin_transformer = None
        # self.categorical_transformer = None
        # self.variate_transformer = None

    def fit(self, X, y=None, *args, **kwargs):
        """ Fit chain ladder.

        Arguments:
        -----------
            X: Numpy array. First column must be origin date. All other columns
               consist of two triangles - count followed by paid, with NaN
               representing bottom triangle.
        """

        if isinstance(X, kl.DataFrame):
            tri = X
        else:
            X = check_array(X, dtype=None, force_all_finite=False)
            tri = pd.DataFrame(X)

        # Incremental triangle
        tri_incr = tri.groupby([0]).apply(lambda g: g.astype(float).sum(skipna=False))

        count_incr = tri_incr.iloc[:, 0 : int((X.shape[1] - 1) / 2)].reset_index()
        paid_incr = tri_incr.iloc[:, int((X.shape[1] - 1) / 2) :]

        super(PPCIRegressor, self).fit(count_incr)

        ultimate_counts = super(PPCIRegressor, self).predict(count_incr).sum(axis=1)

        ppci = paid_incr.div(ultimate_counts, axis=0)

        self.ppci_mean = ppci.agg("mean")

        # Finished fitting!
        self.is_fitted_ = True

    def get_datasets_format(self):
        return (
            ["origin", "claim_count", "claim_paid"],
            ["claim_count", "claim_paid"],
            None,
            None,
        )

    def predict(self, X, y=None, w=None, output="projection"):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, kl.DataFrame):
            tri = X
            pd0 = kl
        else:
            pd0 = pd
            X = check_array(X, dtype=None, force_all_finite=False)
            tri = pd.DataFrame(X)

        count = tri.iloc[:, 0 : int((X.shape[1] - 1) / 2) + 1]

        tri = tri.set_index(tri.columns[0]).astype(float)

        paid = tri.iloc[:, int((X.shape[1] - 1) / 2) :]

        projection_counts = super(PPCIRegressor, self).predict(
            count.to_numpy(), output="projection"
        )

        ultimate_counts = pd.Series(np.sum(projection_counts, axis=1))

        proj_paid = pd0.concat(
            [
                self.ppci_mean.iloc[i] * ultimate_counts
                for i in range(0, self.ppci_mean.shape[0])
            ],
            axis=1,
        )
        proj_paid.columns = paid.columns
        proj_paid.index = paid.index

        paid[paid.isna()] = proj_paid[paid.isna()]

        # Ultimate
        if output == "ultimates_grouped":
            tri_cumu = paid.groupby(paid.index).agg("sum").cumsum(axis=1)

            ultimate_paids = tri_cumu.iloc[:, -1]
            return ultimate_paids
        elif output == "ultimates":
            return paid.cumsum(axis=1).iloc[:, -1]
        elif output == "projection":
            return np.hstack([projection_counts, paid.to_numpy()])

    def score(self, X, y=None):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, kl.DataFrame):
            pd0 = kl
        else:
            pd0 = pd
        X = check_array(X, force_all_finite=False)
        y = check_array(y)

        # Claim paids
        Xtri = pd0.DataFrame(self.predict(X)).iloc[X.shape[1] / 2 :]
        ytri = pd0.DataFrame(y).iloc[X.shape[1] / 2 :]

        # Incremental triangle
        Xtri_incr = tri.groupby([0]).apply(lambda g: g.sum(skipna=False))
        ytri_incr = tri.groupby([0]).apply(lambda g: g.sum(skipna=False))

        return np.sum((ytri_incr.to_numpy() - Xtri_incr.to_numpy()) ** 2)
