import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
import re


def data_as_at(df, as_at_date, column_prefix, periods="months"):
    """ Censors a dataset into a triangle.
    """
    df = df.copy()
    for col in df.columns.tolist():
        if col.startswith(column_prefix):
            periods_to_null = int(re.search(r"\d+$", col).group())

            if periods == "months":
                max_date = as_at_date + relativedelta(months=-periods_to_null)
            elif periods == "days":
                max_date = as_at_date + datetime.timedelta(days=-periods_to_null)
            elif periods == "years":
                max_date = as_at_date + datetime.timedelta(years=-periods_to_null)

            df.loc[df.origin_date > max_date, col] = np.nan

    return df


def benchmark(ds, model, ds_true):
    """ Benchmark one model against another """
    # Get true paid triangle
    true_paid = (
        ds_true.claim_paid.assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )
    ppci_paid = ds.ppci(output="projection")
    model_paid = ds.paid_model(model, output="projection")

    mse_ppci = ((ppci_paid.values - true_paid.values) ** 2).mean(axis=0)
    mse_model = ((model_paid.values - true_paid.values) ** 2).mean(axis=0)

    # print(mse_ppci)
    # print(mse_model)
    return pd.DataFrame(
        np.vstack([mse_ppci, mse_model]),
        index=["PPCI", "Model"],
        columns=true_paid.columns,
    )
