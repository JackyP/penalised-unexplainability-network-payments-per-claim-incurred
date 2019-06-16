import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .plot import plot_triangle
from dateutil.relativedelta import relativedelta
import datetime
import re

try:
    import databricks.koalas as kl
except ImportError:
    print("Optionally, pip install koalas[spark] to use with pyspark.")
    import pandas as kl

    pass


# Convert Strings to Categories


class InsuranceDataset(Dataset):
    """Insurance dataset for actuarial modelling.


    """

    def __init__(
        self,
        features=None,
        origin=None,
        exposure=None,
        claim_count=None,
        claim_paid=None,
        claim_incurred=None,
        as_at_date=None,
        period_type="months",
    ):
        """
        Arguments:
        -----------
            features: Koalas Dataframe or Pandas DataFrame
            origin: Koalas Dataframe or Pandas Series
            delay: Koalas Dataframe or Pandas Series
            exposure: Koalas Dataframe or Pandas Series
            claim_count: Koalas Dataframe or Pandas Dataframe
            claim_paid: Koalas Dataframe or Pandas Dataframe
            claim_incurred: Koalas Dataframe or Pandas Dataframe
            as_at_date: Datetime
        """

        self.features = self.__process_init_frame__(features)
        self.origin = self.__process_init_series__(origin)
        self.exposure = self.__process_init_series__(exposure)
        self.claim_count = self.__data_as_at__(
            self.__process_init_frame__(claim_count),
            self.origin,
            as_at_date,
            period_type,
        )
        self.claim_paid = self.__data_as_at__(
            self.__process_init_frame__(claim_paid),
            self.origin,
            as_at_date,
            period_type,
        )
        self.claim_incurred = self.__data_as_at__(
            self.__process_init_frame__(claim_incurred),
            self.origin,
            as_at_date,
            period_type,
        )
        self.as_at_date = as_at_date
        self.period_type = period_type

    def __converter__(x):
        if x.dtype == "object":
            return x.astype("category")
        elif x.dtype == "int64":
            return x.astype("float32")
        elif x.dtype == "float64":
            return x.astype("float32")
        else:
            return x

    def __data_as_at__(self, df, origin_date, as_at_date, period_type):
        """ Censors a dataset into a triangle.
        """
        if df is None:
            return None

        df.fillna(0, inplace=True)
        if as_at_date is not None:
            for col in df.columns.tolist():
                # if col.startswith(column_prefix):
                periods_to_null = int(re.search(r"\d+$", col).group())

                if period_type == "months":
                    max_date = as_at_date + relativedelta(months=-periods_to_null)
                elif period_type == "days":
                    max_date = as_at_date + datetime.timedelta(days=-periods_to_null)
                elif period_type == "years":
                    max_date = as_at_date + datetime.timedelta(years=-periods_to_null)

                df.loc[origin_date > max_date, col] = np.nan
        return df

    def __process_init_frame__(self, df):
        """ Attempt to coerce to DataFrame
        """
        if df is None:
            return None
        elif isinstance(df, pd.DataFrame):
            return df
        elif isinstance(df, kl.DataFrame):
            return df
        else:
            return pd.DataFrame(df)

    def __process_init_series__(self, s):
        """ Attempt to coerce to right format
        """
        if s is None:
            return None
        else:
            return self.__process_init_frame__(s).iloc[:, 0]  # Currently same process

    def __len__(self):
        """ Pytorch - number of items """
        return self.origin.count() + 0

    def __getitem__(self, idx):
        """ Pytorch - get item by index """
        return "TODO"

    def project_claim_count(self, model, future_only=True):
        """ Projects the claim count triangle """

        X = self.X(model)
        y = self.y(model)
        w = self.w(model)

        y_format = model.get_datasets_format()[1]
        ind = y_format.index("claim_count")
        veclen = len(self.claim_count.columns)

        if w is None:
            predict = model.predict(X, y)
        else:
            predict = model.predict(X, y, w=w)

        if isinstance(self.claim_paid, pd.DataFrame):
            predict = pd.DataFrame(
                predict[:, ind * veclen : (ind + 1) * veclen],
                index=self.claim_count.index,
                columns=self.claim_count.columns,
            )
        elif isinstance(self.claim_paid, kl.DataFrame):
            predict = kl.DataFrame(
                predict[:, ind * veclen : (ind + 1) * veclen],
                index=self.claim_count.index,
                columns=self.claim_count.columns,
            )

        if future_only:
            claim_count = self.claim_count.fillna(predict)
        else:
            claim_count = predict

        return claim_count

    def plot_triangle_claim_count(self, model=None):
        """ Plots the claim count triangle """
        if model is not None:
            claim_count = self.project_claim_count(model)
            mask_bottom = False
        else:
            claim_count = self.claim_count
            mask_bottom = True

        return plot_triangle(
            claim_count.assign(origin_date=self.origin)
            .groupby(["origin_date"])
            .agg("sum")
            .cumsum(axis=1),
            mask_bottom,
        )

    def project_claim_paid(self, model, future_only=True):
        """ Predict the claim paid triangle """

        X = self.X(model)
        y = self.y(model)
        w = self.w(model)

        y_format = model.get_datasets_format()[1]
        ind = y_format.index("claim_paid")
        veclen = len(self.claim_paid.columns)

        if w is None:
            predict = model.predict(X, y)
        else:
            predict = model.predict(X, y, w=w)

        if isinstance(predict, pd.DataFrame):
            predict = predict.iloc[:, ind * veclen : (ind + 1) * veclen]
        elif isinstance(predict, kl.DataFrame):
            predict = predict.iloc[:, ind * veclen : (ind + 1) * veclen]
        elif isinstance(self.claim_paid, pd.DataFrame):
            predict = pd.DataFrame(
                predict[:, ind * veclen : (ind + 1) * veclen],
                index=self.claim_paid.index,
                columns=self.claim_paid.columns,
            )
        elif isinstance(self.claim_paid, kl.DataFrame):
            predict = kl.DataFrame(
                predict[:, ind * veclen : (ind + 1) * veclen],
                index=self.claim_paid.index,
                columns=self.claim_paid.columns,
            )

        if future_only:
            claim_paid = self.claim_paid.fillna(predict)
        else:
            claim_paid = predict
        return claim_paid

    def plot_triangle_claim_paid(self, model=None):
        """ Plots the claim paid triangle """
        if model is not None:
            claim_paid = self.project_claim_paid(model)
            mask_bottom = False
        else:
            claim_paid = self.claim_paid
            mask_bottom = True

        return plot_triangle(
            claim_paid.assign(origin_date=self.origin)
            .groupby(["origin_date"])
            .agg("sum")
            .cumsum(axis=1),
            mask_bottom,
        )

    def w(self, Model):
        """ weights for scikit-learn estimators
        """
        if self.exposure is None:
            return np.ones(self.claim_paid.shape[0]).reshape(-1, 1)
        else:
            return self.exposure

    def X(self, Model):
        """ X for scikit-learn estimators
        """

        input_X_datasets, input_y_datasets, preprocess_X, preprocess_y = (
            Model.get_datasets_format()
        )

        datasets = [getattr(self, dataset_name) for dataset_name in input_X_datasets]

        if isinstance(datasets[0], pd.DataFrame) or isinstance(datasets[0], pd.Series):
            df = pd.concat(datasets, axis=1)
        else:
            df = kl.concat(datasets, axis=1)

        if preprocess_X is None:
            return df.to_numpy()
        else:
            return preprocess_X(df)

    def y(self, Model):
        """ y for scikit-learn estimators
        """
        input_X_datasets, input_y_datasets, preprocess_X, preprocess_y = (
            Model.get_datasets_format()
        )

        if input_y_datasets is None:
            datasets = None
        else:
            datasets = [
                getattr(self, dataset_name) for dataset_name in input_y_datasets
            ]

        if isinstance(self.claim_count, pd.DataFrame):
            return pd.concat(datasets, axis=1).to_numpy()
        else:
            return kl.concat(datasets, axis=1).to_numpy()
