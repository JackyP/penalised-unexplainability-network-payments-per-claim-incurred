import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
import re


def fetch_example_data(nrows=10000):
    """ Example synthetic dataset based on the open airline delay dataset
    """

    df = (
        pd.read_csv(
            "https://raw.githubusercontent.com/yankev/testing/master/datasets/nycflights.csv",
            # nrows=nrows,
        )
        .assign(
            origin_date=lambda df: pd.to_datetime(df[["year", "month", "day"]])
            + MonthEnd(1),
            ultimate_claim_count=lambda df: np.where(df.arr_delay > 0, 1, 0),
            ultimate_claim_size=lambda df: np.where(
                df.arr_delay > 0, df.arr_delay * 5000, 0
            ),
            expected_delay=lambda df: np.abs(np.floor(df.dep_delay / 3)),
        )
        .rename(index=str, columns={"origin": "departing", "dest": "destination"})[
            [
                "origin_date",
                "carrier",
                "flight",
                "departing",
                "destination",
                "distance",
                "ultimate_claim_count",
                "ultimate_claim_size",
                "expected_delay",
            ]
        ]
        .sample(nrows, random_state=12345)
    )

    # Make up the claim payments
    for i in range(0, 11):
        df["claim_count_{}".format(i)] = np.where(
            df["expected_delay"] == i, df.ultimate_claim_count, 0
        )

    # Make up the claim payments
    for i in range(0, 11):
        df["claim_paid_{}".format(i)] = np.where(
            df["expected_delay"] == i, df.ultimate_claim_size, 0
        )

    df["claim_count_11"] = np.where(
        df["expected_delay"] >= 12, df.ultimate_claim_count, 0
    )

    df["claim_paid_11"] = np.where(
        df["expected_delay"] >= 12, df.ultimate_claim_size, 0
    )

    return df


if __name__ == "__main__":
    # This code can be removed later
    df = fetch_example_data()
    df.groupby(["origin_date"]).agg("sum")

    tri = (
        data_as_at(df, df.origin_date.agg("max"), "claim_count")
        .groupby(["origin_date"])
        .agg("sum")
    )

    tri_only = tri.loc[:, tri.columns.str.startswith("claim_count")]

    tri_only
