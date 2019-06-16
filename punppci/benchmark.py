import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .dataset import InsuranceDataset
import gc


def benchmark(ds, ds_true, model0, model1):
    """ Benchmark one model against another """
    # Get true count triangle
    true_count = (
        ds_true.claim_count.assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )
    model1_count = (
        ds.project_claim_count(model1)
        .assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )
    model0_count = (
        ds.project_claim_count(model0)
        .assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )

    mse_model1_count = ((model1_count.to_numpy() - true_count.to_numpy()) ** 2).mean(
        axis=0
    )
    mse_model0_count = ((model0_count.to_numpy() - true_count.to_numpy()) ** 2).mean(
        axis=0
    )

    count_loss = pd.DataFrame(
        np.vstack([mse_model0_count, mse_model1_count]),
        index=[type(model0).__name__, type(model1).__name__],
        columns=true_count.columns,
    )

    # Get true paid triangle
    true_paid = (
        ds_true.claim_paid.assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )

    model1_paid = (
        ds.project_claim_paid(model1)
        .assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )
    model0_paid = (
        ds.project_claim_paid(model0)
        .assign(origin_date=ds_true.origin)
        .groupby(["origin_date"])
        .agg("sum")
        .cumsum(axis=1)
    )

    mse_model0 = ((model0_paid.to_numpy() - true_paid.to_numpy()) ** 2).mean(axis=0)
    mse_model1 = ((model1_paid.to_numpy() - true_paid.to_numpy()) ** 2).mean(axis=0)

    paid_loss = pd.DataFrame(
        np.vstack([mse_model0, mse_model1]),
        index=[type(model0).__name__, type(model1).__name__],
        columns=true_paid.columns,
    )

    true_paid["model"] = "True"
    model0_paid["model"] = type(model0).__name__
    model1_paid["model"] = type(model1).__name__

    # print(mse_ppci)
    # print(mse_model)
    return (
        pd.concat([count_loss, paid_loss], axis=1),
        pd.concat([true_paid, model0_paid, model1_paid], axis=0),
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
        weights_model = ds_true.w(model)
        weights_actual = ds_true.w(model)

        pred = (
            ds_true.project_claim_count(model, future_only=False)
            .sum(axis=1)
            .reset_index(drop=True)
        )

    elif output == "size":
        # Actual vs Model Claim Size, Weights = Claim Counts
        actual = ds_true.claim_paid.reset_index(drop=True)
        weights_model = (
            ds_true.project_claim_count(model, future_only=False)
            .sum(axis=1)
            .reset_index(drop=True)
        )

        weights_actual = ds_true.claim_count.reset_index(drop=True).sum(axis=1)

        pred = (
            ds_true.project_claim_paid(model, future_only=False)
            .sum(axis=1)
            .reset_index(drop=True)
        )
    elif output == "cost":
        # Actual vs Model Claim Paid, Weights = Claim Counts
        actual = ds_true.claim_paid.reset_index(drop=True)
        weights_model = ds_true.w(model)
        weights_actual = ds_true.w(model)

        pred = (
            ds_true.project_claim_paid(model, future_only=False)
            .sum(axis=1)
            .reset_index(drop=True)
        )
    else:
        raise Exception("Please either have output = 'frequency', 'size' or 'paid'.")

    # Do we need to bin
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
    weights_model = ds_true.w(model)
    weights_actual = ds_true.w(model)

    pred = (
        ds_true.project_claim_paid(model, future_only=False)
        .sum(axis=1)
        .reset_index(drop=True)
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
    Model0,
    Model1,
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

    if n > 0:
        df = df.sample(n=n, random_state=v)

    if exposure is None:
        exposure_df = None
    else:
        exposure_df = df[exposure]

    # Dataset
    ds = InsuranceDataset(
        features=df[features],
        origin=df[origin],
        exposure=exposure_df,
        claim_count=df[claim_count],
        claim_paid=df[claim_paid],
        as_at_date=as_at,
    )

    ds_true = InsuranceDataset(
        features=df[features],
        origin=df[origin],
        exposure=exposure_df,
        claim_count=df[claim_count],
        claim_paid=df[claim_paid],
        as_at_date=None,
    )

    # Model - Make, Fit and Predict

    model0 = Model0()

    model0.fit(ds.X(model0), ds.y(model0), w=ds.w(model0))

    model1 = Model1()

    model1.fit(ds.X(model1), ds.y(model1), w=ds.w(model1))

    # Explained vs Unexplained factors
    bench_mod_reserve, paid = benchmark(ds, ds_true, model0, model1)
    bench_mod_price = actual_vs_model_predicted_cost(ds_true, model0)

    print("------------------------------------------------------------------")
    print("For test case #{} with n={}".format(v, n))

    print("PPCI vs Model Errors of:")
    print(bench_mod_reserve)
    print("Loss Prediction Comparison:")
    print(bench_mod_price)

    # Attach data
    bench_mod_reserve = bench_mod_reserve.assign(n=n, v=v)
    bench_mod_price = bench_mod_price.assign(n=n, v=v)
    paid = paid.assign(n=n, v=v)
    gc.collect()

    return model0, model1, paid, bench_mod_reserve, bench_mod_price


def benchmark_test_suite(
    df,
    Model0,
    Model1,
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
            Model0,
            Model1,
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

    models0 = [a0 for a0, a1, b, c, d in results]
    models1 = [a1 for a0, a1, b, c, d in results]
    paid = pd.concat([b for a0, a1, b, c, d in results])
    bench_mod_reserve = pd.concat([c for a0, a1, b, c, d in results])
    bench_mod_price = pd.concat([d for a0, a1, b, c, d in results])

    return models0, models1, paid, bench_mod_reserve, bench_mod_price
