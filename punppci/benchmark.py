import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import re
from .dataset import Dataset
from .pytorch import PUNPPCIClaimRegressor
import gc


def data_as_at(df, origin_date, as_at_date, column_prefix, periods="months"):
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

            df.loc[df[origin_date] > max_date, col] = np.nan

    return df


def benchmark(ds, model, ds_true):
    """ Benchmark one model against another """
    # Get true count triangle
    true_count = (
        ds_true.claim_count.assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )
    chain_ladder_count = ds.chain_ladder_count(output="projection")
    model_count = ds.count_model(model, output="projection")

    mse_chain_ladder_count = (
        (chain_ladder_count.values - true_count.values) ** 2
    ).mean(axis=0)
    mse_model_count = ((model_count.values - true_count.values) ** 2).mean(axis=0)

    count_loss = pd.DataFrame(
        np.vstack([mse_chain_ladder_count, mse_model_count]),
        index=["PPCI", "Model"],
        columns=true_count.columns,
    )

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

    paid_loss = pd.DataFrame(
        np.vstack([mse_ppci, mse_model]),
        index=["PPCI", "Model"],
        columns=true_paid.columns,
    )

    true_paid["model"] = "True"
    ppci_paid["model"] = "PPCI"
    model_paid["model"] = "Model"

    # print(mse_ppci)
    # print(mse_model)
    return (
        pd.concat([count_loss, paid_loss], axis=1),
        pd.concat([true_paid, ppci_paid, model_paid], axis=0),
    )


def actual_vs_model(ds_true, model, factor, output="cost", num_bands=20):
    """ Compare actual vs model
    Parameters:
    -----------
    ds_true: punppci.Dataset
    model: punppci. or PUNPPCILossOptimizer model
    factor: Name of a feature
    output: "frequency" or "size"
    no_bands: Number of bands
    """
    if output == "frequency":
        # Actual vs Model Claim Count, Weights = Weights
        actual = ds_true.claim_count.reset_index(drop=True)
        weights_model = ds_true.w()
        weights_actual = ds_true.w()

    elif output == "size":
        # Actual vs Model Claim Size, Weights = Claim Counts
        actual = ds_true.claim_paid.reset_index(drop=True)
        freq_cols = ds_true.claim_count.columns.tolist()
        weights_model = model.predict(ds_true.X())[freq_cols].sum(axis=1)
        weights_actual = ds_true.claim_count.reset_index(drop=True).sum(axis=1)
    elif output == "cost":
        # Actual vs Model Claim Paid, Weights = Claim Counts
        actual = ds_true.claim_paid.reset_index(drop=True)
        weights_model = ds_true.w()
        weights_actual = ds_true.w()
    else:
        raise Exception("Please either have output = 'frequency', 'size' or 'paid'.")

    # Do we need to bin
    cols = actual.columns.tolist()

    pred = (
        model.predict(ds_true.X())[cols]
        .sum(axis=1)
        .multiply(pd.Series(ds_true.w()), axis=0)
    )

    factor_s = ds_true.features[factor].reset_index(drop=True)

    if (
        (factor_s.drop_duplicates().shape[0] > num_bands)
        and (factor_s.dtype == np.float)
        or (factor_s.dtype == np.int)
    ):
        factor_int, bins = pd.qcut(
            factor_s, num_bands, retbins=True, labels=False, duplicates="drop"
        )
        factor_s = bins[1:][factor_int]

    # Combine
    combined = (
        pd.concat([pred, actual.sum(axis=1)], axis=1)
        .assign(
            factor=factor_s, Weights_Model=weights_model, Weights_Actual=weights_actual
        )
        .groupby("factor")
        .agg("sum")
    )

    combined[0] = combined[0] / combined["Weights_Model"]
    combined[1] = combined[1] / combined["Weights_Actual"]

    combined.columns = [
        "Model {}".format(output.title()),
        "Actual {}".format(output.title()),
        "Weights Model",
        "Weights Actual",
    ]

    return combined


def actual_vs_model_predicted_cost(ds_true, model, num_bands=20):
    """ Plot actual vs model by predicted_cost """
    # Actual vs Model Claim Paid, Weights = Claim Counts
    actual = ds_true.claim_paid.reset_index(drop=True)
    weights_model = ds_true.w()
    weights_actual = ds_true.w()

    cols = actual.columns.tolist()

    pred = (
        model.predict(ds_true.X())[cols]
        .sum(axis=1)
        .multiply(pd.Series(ds_true.w()), axis=0)
    )

    factor_s = pred

    try:
        factor_int, bins = pd.qcut(
            factor_s, num_bands, retbins=True, labels=False, duplicates="drop"
        )
        factor_s = bins[1:][factor_int]
    except:
        pass

    # Combine
    combined = (
        pd.concat([pred, actual.sum(axis=1)], axis=1)
        .assign(
            factor=factor_s, Weights_Model=weights_model, Weights_Actual=weights_actual
        )
        .groupby("factor")
        .agg("sum")
    )

    combined[0] = combined[0] / combined["Weights_Model"]
    combined[1] = combined[1] / combined["Weights_Actual"]

    combined.columns = ["Model Paid", "Actual Paid", "Weights Model", "Weights Actual"]

    return combined


def plot_actual_vs_model(ds_true, model, factor, num_bands=20):
    """ Plot actual vs model
    Parameters:
    -----------
    ds_true: punppci.Dataset
    model: punppci. or PUNPPCILossOptimizer model
    factor: Name of a feature
    num_bands: number of bands
    """
    plt.figure(figsize=(7, 11))

    # Freq and Size
    freq = actual_vs_model(
        ds_true, model, factor, output="frequency", num_bands=num_bands
    )
    size = actual_vs_model(ds_true, model, factor, output="size", num_bands=num_bands)

    # Subplots
    c = pd.concat([freq, size], axis=1)
    fig, axs = plt.subplots(2, 1)

    # Matplotlib cannot handle categoricals
    if isinstance(freq.index.dtype, pd.CategoricalDtype):
        c.index = c.index.astype(str)

    axs[0].plot(c.index, c["Model Frequency"], c.index, c["Actual Frequency"])
    axs[0].set_xlabel(factor)
    axs[0].set_ylabel("Frequency")
    axs[0].grid(True)

    axs[1].plot(c.index, c["Model Size"], c.index, c["Actual Size"])
    axs[1].set_xlabel(factor)
    axs[1].set_ylabel("Size")

    return fig, axs


def benchmark_tester(
    df,
    Model,
    n,
    v=0,
    features=None,
    origin=None,
    exposure=None,
    claim_count=None,
    claim_paid=None,
    claim_prefix=None,
    as_at=None,
    export_csv=False,
):
    """ PUNPPCI - does it work?


    """
    prefix = f"{n}_test{v}_"

    df_full = df.sample(n=n, random_state=v)

    df = data_as_at(df_full, origin, as_at, claim_prefix)

    if exposure is None:
        exposure_df = None
    else:
        exposure_df = df[exposure]

    # Dataset
    # A smarter normalisation without data leakage is preferable, but this is sufficient
    # for v1

    ds_orig = Dataset(
        features=df[features],
        origin=df[origin],
        exposure=exposure_df,
        claim_count=df[claim_count],
        claim_paid=df[claim_paid],
    )

    avg_count = ds_orig.chain_ladder_count(output="ultimates").sum() / ds_orig.w().sum()
    avg_paid = ds_orig.ppci(output="ultimates").sum() / ds_orig.w().sum()

    print(avg_count, avg_paid)

    ds = Dataset(
        features=df[features],
        origin=df[origin],
        exposure=exposure_df,
        claim_count=df[claim_count].divide(avg_count),
        claim_paid=df[claim_paid].divide(avg_paid),
    )

    ds_true = Dataset(
        features=df_full[features],
        origin=df_full[origin],
        exposure=exposure_df,
        claim_count=df_full[claim_count].divide(avg_count),
        claim_paid=df_full[claim_paid].divide(avg_paid),
    )

    # Model - Make, Fit and Predict
    # model = PUNPPCILossEstimator(dataset=ds)
    # model.fit(ds.X(), ds.y(), w=ds.w(), verbose=0, batch_size=ds.X().shape[0])

    model = Model(
        feature_dimension=29,
        output_dimension=df[claim_count].shape[1],
        claim_count_names=claim_count,
        claim_paid_names=claim_paid,
    )

    model.fit(ds.X(), ds.y())

    # Explained vs Unexplained factors
    # lin_vs_res = model.linear_vs_residual()

    ppci_acs = ds.ppci(output="selections").sum()
    model_acs = ds.payments_per_claim_model(model).sum()

    # mean_frequency = np.exp(model.claim_count_initializer)

    # model_frequency = model.predict(ds.X(), "frequency").mean()

    bench_mod_reserve, paid = benchmark(ds, model, ds_true)
    bench_mod_price = actual_vs_model_predicted_cost(ds_true, model)

    print("------------------------------------------------------------------")
    print("For test case #{} with n={}".format(v, n))
    print("PPCI Average Claim Size: {} vs Model: {}".format(ppci_acs, model_acs))
    # print("Average frequency: {} vs Model: {}".format(mean_frequency, model_frequency))
    print("Linear vs residual of:")
    # print(lin_vs_res)
    print("PPCI vs Model Errors of:")
    print(bench_mod_reserve)
    print("Loss Prediction Comparison:")
    print(bench_mod_price)

    # weights = model.get_weights()
    # Bias check
    """
    print("Bias check:")
    print(
        "Ultimate Count:",
        np.exp(weights["risk_count_linear_output_ultimate_claim_count/bias:0"]),
        np.exp(weights["risk_count_residual_output_ultimate_claim_count/bias:0"]),
        "Ultimate Size:",
        np.exp(weights["risk_size_linear_output_ultimate_claim_size/bias:0"]),
        np.exp(weights["risk_size_residual_output_ultimate_claim_size/bias:0"]),
        "Development Count:",
        weights["develop_count_residual_output_claim_count/bias:0"],
        weights["develop_count_linear_output_claim_count/bias:0"],
        "Development Size:",
        weights["develop_count_residual_output_claim_count/bias:0"],
        weights["develop_count_linear_output_claim_count/bias:0"],
    )
    """
    # Output Counts - to CSV
    if export_csv:
        bench_mod_reserve.to_csv(f"output/{prefix}_benchmark_loss_reserve_results.csv")
        bench_mod_price.to_csv(f"output/{prefix}_benchmark_loss_price_results.csv")

        # lin_vs_res.to_csv(f"output/{prefix}_linear_vs_residual.csv")

        ds.chain_ladder_count(output="projection").to_csv(
            f"output/{prefix}_count_ppci.csv"
        )
        ds.count_model(model, output="projection").to_csv(
            f"output/{prefix}_count_model.csv"
        )
        ds_true.claim_count.assign(origin_date=ds_true.origin).groupby(
            ["origin_date"]
        ).agg("sum").cumsum(axis=1).to_csv(f"output/{prefix}_count_true.csv")

        # Output Paid - to CSV
        ds.ppci(output="projection").to_csv(f"output/{prefix}_paid_ppci.csv")
        ds.paid_model(model, output="projection").to_csv(
            f"output/{prefix}_paid_model.csv"
        )
        ds_true.claim_paid.assign(origin_date=ds_true.origin).groupby(
            ["origin_date"]
        ).agg("sum").cumsum(axis=1).to_csv(f"output/{prefix}_paid_true.csv")

    # Attach data
    # lin_vs_res = lin_vs_res.assign(n=n, v=v)
    bench_mod_reserve = bench_mod_reserve.assign(n=n, v=v)
    bench_mod_price = bench_mod_price.assign(n=n, v=v)
    paid = paid.assign(n=n, v=v)
    gc.collect()

    return model, paid, bench_mod_reserve, bench_mod_price


def benchmark_test_suite(
    df,
    Model,
    n_list=[
        5000,
        5000,
        5000,
        5000,
        5000,
        10000,
        10000,
        10000,
        10000,
        10000,
        25000,
        25000,
        25000,
        25000,
        25000,
        50000,
        50000,
        50000,
        50000,
        50000,
        100000,
        200000,
    ],  #
    v_list=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 1],  #
    features=None,
    origin=None,
    exposure=None,
    claim_count=None,
    claim_paid=None,
    claim_prefix=None,
    as_at=None,
):
    """ PUNPPCI - does it work?


    """
    results = [
        benchmark_tester(
            df,
            Model,
            n,
            v,
            features=features,
            origin=origin,
            exposure=exposure,
            claim_count=claim_count,
            claim_paid=claim_paid,
            claim_prefix=claim_prefix,
            as_at=as_at,
        )
        for n, v in zip(n_list, v_list)
    ]

    models = [a for a, b, c, d in results]
    paid = pd.concat([b for a, b, c, d in results])
    bench_mod_reserve = pd.concat([c for a, b, c, d in results])
    bench_mod_price = pd.concat([d for a, b, c, d in results])

    return models, paid, bench_mod_reserve, bench_mod_price
