import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn import preprocessing
from .plot import plot_triangle

"""
    Outstanding items:
     - Dask
     - Generator
     - Types of models
"""


class Dataset:
    """
        Convenience class for data transformation for neural network models of
        structured data

        Properties
        ----------
        features
        origin
        delay

        exposure

        claim_count
        claim_paid

        variate_names
        variate_scaler

        category_names
        category_levels

        Methods
        -------
        __init__(self, data, exposure_weight, delay, claim_count, cumulative_paid):

        X
        y


    """

    def __init__(
        self,
        features=None,
        origin=None,
        exposure=None,
        claim_count=None,
        claim_paid=None,
    ):
        """ Create a dataset for actuarial modelling

        Parameters:
        -----------
        features: Pandas DataFrame
        origin: Pandas Series
        delay: Pandas Series
        exposure: Pandas Series
        claim_count: Pandas Dataframe
        claim_paid: Pandas Dataframe
        """

        # Data validation
        try:
            # Check each column
            for col, col_type in zip(
                [features, origin, exposure, claim_count, claim_paid],
                [
                    "features",
                    "origin",
                    "exposure",
                ],  # don't null check "claim count", "claim paid"
            ):
                if col is not None:
                    assert (
                        # Check null values
                        col.isnull().values.any()
                        == 0
                    ), "Dataframe contains null values! Please clean these values before creating the Dataset"
                else:
                    if col_type == "exposure":
                        # Create dummy vector with same length as origin
                        exposure = pd.DataFrame(
                            1.0, index=np.arange(len(origin)), columns=["exposure"]
                        )["exposure"]
                    else:
                        raise AssertionError(
                            "Data for {} not optional!".format(col_type)
                        )
        except (AttributeError, TypeError):
            raise AssertionError("Input data should be a Pandas DataFrame!")

        # Convert Strings to Categories
        def converter(x):
            if x.dtype == "object":
                return x.astype("category")
            elif x.dtype == "int64":
                return x.astype("float64")
            else:
                return x

        features_convert = features.apply(converter)

        features_convert["origin"] = origin.values.astype(float)

        # Transformer - Features
        categorical_features = features_convert.dtypes == "category"
        origin_features = features_convert.columns == "origin"
        numerical_features = ~(categorical_features | origin_features)

        self.preprocess = make_column_transformer(
            (preprocessing.MinMaxScaler(), origin_features),
            (preprocessing.OrdinalEncoder(), categorical_features),
            (preprocessing.StandardScaler(), numerical_features),
        )

        self.preprocess.fit(features_convert)

        # Transformer - Responses

        # Store attributes
        self.category_names = features_convert.columns[categorical_features].tolist()
        self.variate_names = features_convert.columns[numerical_features].tolist()

        self.category_levels = [
            features_convert[x].cat.categories.shape[0] for x in self.category_names
        ]

        self.features = features_convert
        self.feature_names = ["origin"] + self.category_names + self.variate_names
        self.origin = origin
        self.exposure = exposure
        self.claim_count = claim_count
        self.claim_paid = claim_paid

    def chain_ladder_count(self, output="ultimates"):
        """ Estimate ultimate claim counts based on chain ladder """
        # Claim numbers
        tri = self.claim_count.copy()
        tri["origin_date"] = self.origin

        # Incremental triangle
        tri_incr = tri.groupby(["origin_date"]).apply(lambda g: g.sum(skipna=False))

        # Cumulative triangle
        tri_cumu = tri_incr.cumsum(axis=1)

        # Chain Ladder Ratios
        for i in range(0, tri_incr.shape[1] - 1):
            isvalid = ~np.isnan(tri_cumu.iloc[:, i + 1])
            chain_ladder_factor = tri_cumu.iloc[:, i + 1].loc[isvalid].agg(
                "sum"
            ) / tri_cumu.iloc[:, i].loc[isvalid].agg("sum")

            # Apply development
            tri_cumu.iloc[:, i + 1].loc[~isvalid] = (
                tri_cumu.iloc[:, i].loc[~isvalid] * chain_ladder_factor
            )
            # print(chain_ladder_factor)

        # Ultimate
        if output == "ultimates":
            ultimate_counts = tri_cumu.iloc[:, -1]
            return ultimate_counts

        elif output == "projection":
            return tri_cumu

    def paid_model(self, model, output="ultimates"):
        """ Apply a paid model and return ultimates or data """
        # Weighted predictions
        proj_paid_detail = model.predict(self.X())

        proj_paid_detail = proj_paid_detail.multiply(pd.Series(self.w()), axis=0)

        proj_paid_detail["origin_date"] = self.origin.values

        proj_paid = proj_paid_detail.groupby(["origin_date"]).apply(
            lambda g: g.sum(skipna=False)
        )

        # Fill
        tri = self.claim_paid.copy()
        tri["origin_date"] = self.origin

        # Incremental triangle
        tri_incr = tri.groupby(["origin_date"]).apply(lambda g: g.sum(skipna=False))
        tri_incr[tri_incr.isna()] = proj_paid[tri_incr.isna()]

        # Cumulative triangle
        tri_cumu = tri_incr.cumsum(axis=1)

        # Ultimate
        if output == "ultimates":
            ultimate_paids = tri_cumu.iloc[:, -1]
            return ultimate_paids

        elif output == "projection":
            return tri_cumu

    def plot_triangle_claim_count(self, mask_bottom=True):
        """ Plots the claim count triangle """
        claim = self.claim_count.copy()
        claim["origin_date"] = self.origin

        return plot_triangle(
            claim.groupby(["origin_date"]).agg("sum").cumsum(axis=1), mask_bottom
        )

    def plot_triangle_claim_paid(self, mask_bottom=True):
        """ Plots the claim paid triangle """
        claim = self.claim_paid.copy()
        claim["origin_date"] = self.origin

        return plot_triangle(
            claim.groupby(["origin_date"]).agg("sum").cumsum(axis=1), mask_bottom
        )

    def plot_triangle_ppci(self, mask_bottom=False):
        """ Plots the projected paid triangle using PPCI """
        return plot_triangle(self.ppci(output="projection"), mask_bottom)

    def plot_triangle_model(self, model, mask_bottom=False):
        """ Plots the projected paid triangle using PUNPPCI """
        return plot_triangle(self.paid_model(model, output="projection"), mask_bottom)

    def ppci(self, output="ultimates"):
        """ Payments per claim incurred """
        tri = self.claim_paid.copy()
        tri["origin_date"] = self.origin

        # Incremental triangle
        tri_incr = tri.groupby(["origin_date"]).apply(lambda g: g.sum(skipna=False))

        # PPCI

        ultimate_counts = self.chain_ladder_count()
        ppci = tri_incr.div(ultimate_counts, axis=0)

        ppci_mean = ppci.agg("mean")

        # Projection
        proj_paid = pd.concat(
            [ultimate_counts for x in range(0, ppci_mean.shape[0])], axis=1
        )
        proj_paid.columns = tri_incr.columns
        proj_paid.multiply(ppci_mean, axis=1)

        tri_incr[tri_incr.isna()] = proj_paid[tri_incr.isna()]

        # Cumulative triangle
        tri_cumu = tri_incr.cumsum(axis=1)

        # Ultimate
        if output == "ultimates":
            ultimate_paids = tri_cumu.iloc[:, -1]
            return ultimate_paids

        elif output == "projection":
            return tri_cumu

    def w(self):
        """ Return weights
        """
        return self.exposure.values

    def X(self):
        """ X for scikit-learn estimators
        """
        return self.preprocess.transform(self.features)

    def y(self):
        """ y for scikit-learn estimators
        """
        return pd.concat([self.claim_count, self.claim_paid], axis=1).values
