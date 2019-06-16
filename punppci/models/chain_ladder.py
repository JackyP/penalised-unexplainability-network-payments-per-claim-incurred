from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import numpy as np

try:
    import databricks.koalas as kl
except ImportError:
    import pandas as kl

    pass


class ChainLadderRegressor(BaseEstimator):
    """ Estimate ultimates based on chain ladder """

    def __init__(self, claim_type):
        """ Init """
        self.claim_type = claim_type
        assert claim_type in ["claim_count", "claim_paid", "claim_incurred"]
        # self.input_y_datasets = None
        # self.origin_transformer = None
        # self.categorical_transformer = None
        # self.variate_transformer = None

    def fit(self, X, y=None, *args, **kwargs):
        """ Fit chain ladder.

        Arguments:
        -----------
            X: Numpy array. First column must be origin date. All other columns
               consist of one triangle, with NaN representing bottom triangle.
        """

        if isinstance(X, kl.DataFrame):
            tri = X
        else:
            X = check_array(X, dtype=None, force_all_finite=False)
            # Claim numbers
            tri = pd.DataFrame(X)

        # Incremental triangle
        tri_incr = tri.groupby([0]).apply(lambda g: g.astype(float).sum(skipna=False))

        # Cumulative triangle
        tri_cumu = tri_incr.cumsum(axis=1)

        self.chain_ladder_factors = []
        # Chain Ladder Ratios
        for i in range(0, tri_incr.shape[1] - 1):
            isvalid = ~np.isnan(tri_cumu.iloc[:, i + 1])
            chain_ladder_factor = tri_cumu.iloc[:, i + 1].loc[isvalid].agg(
                "sum"
            ) / tri_cumu.iloc[:, i].loc[isvalid].agg("sum")

            self.chain_ladder_factors += [chain_ladder_factor]

        # Finished fitting!
        self.is_fitted_ = True

    def get_datasets_format(self):
        return (["origin", self.claim_type], [self.claim_type], None, None)

    def predict(self, X, y=None, w=None, output="projection"):
        check_is_fitted(self, "is_fitted_")

        if isinstance(X, kl.DataFrame):
            tri = X

        else:

            X = check_array(X, dtype=None, force_all_finite=False)
            tri = pd.DataFrame(X)

        tri_incr = tri.set_index(0).astype(float)

        # Cumulative triangle
        tri_cumu = tri_incr.cumsum(axis=1)

        for i in range(0, tri_incr.shape[1] - 1):
            # Apply development
            isvalid = ~np.isnan(tri_cumu.iloc[:, i + 1])

            tri_cumu.iloc[:, i + 1].loc[~isvalid] = (
                tri_cumu.iloc[:, i].loc[~isvalid] * self.chain_ladder_factors[i]
            )

            # Recover incremental valuesv
            tri_incr.iloc[:, i + 1] = tri_cumu.iloc[:, i + 1] - tri_cumu.iloc[:, i]

        # Ultimate
        if output == "ultimates_grouped":
            tri_cumu = tri_cumu.groupby(tri_cumu.index).agg("sum")
            ultimate_counts = tri_cumu.iloc[:, -1]
            return ultimate_counts
        elif output == "ultimates":
            return tri_cumu.iloc[:, -1]
        elif output == "projection":
            return tri_incr.to_numpy()

    def score(self, X, y=None):
        check_is_fitted(self, "is_fitted_")
        if isinstance(X, kl.DataFrame):
            pd = kl
        X = check_array(X, dtype=None, force_all_finite=False)
        y = check_array(y)

        # Claim numbers
        Xtri = pd.DataFrame(self.predict(X))
        ytri = pd.DataFrame(y)

        # Incremental triangle
        Xtri_incr = tri.groupby([0]).apply(lambda g: g.sum(skipna=False))
        ytri_incr = tri.groupby([0]).apply(lambda g: g.sum(skipna=False))

        return np.sum((ytri_incr.to_numpy() - Xtri_incr.to_numpy()) ** 2)
