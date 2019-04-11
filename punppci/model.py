from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
from .networks import (
    network_inputs,
    penalised_unexplainability_network,
    network_embeddings,
)
from keras.layers import Lambda, Reshape
from keras.models import Model, Sequential
from keras.utils.vis_utils import model_to_dot
from keras.utils.generic_utils import has_arg
from keras.layers import concatenate, Activation
from keras import backend as K
from keras import optimizers
from IPython.display import display, SVG
from pdpbox import info_plots, pdp
import pandas as pd
import numpy as np


class PUNPPCILossEstimator(BaseEstimator):
    """
        Sci-kit Learn compatible non-life insurance loss estimator

        Properties:
        -----------

        variate_names
        category_names
        origin_name
        delay_name
        exposure_name

        category_levels
        category_dimensions
        variate_scaler

        claim_count_network
        development_network
        kwargs

        model

        Methods:
        --------
        __init__(
            dataset,
            claim_count_network,
            expected_cost_network,
            development_network,
            **kwargs
        )

        fit(
            X, <- array, DataFrame, Dataset, DaskFrame [keras.model.partial_fit]
            y,
            optimizer,
            claim_count_loss='poisson',
            claim_payment_loss='mean_squared_error',
            *args,
            **kwargs
        )

        predict(
            X,
            response,
            *args,
            **kwargs,
        )

        explain()

        ppci()

        plot_network()
        plot_partial()
        plot_actual()
    """

    def __init__(
        self,
        dataset=None,
        # Dataset parameters
        variate_names=None,
        category_names=None,
        category_levels=None,
        feature_names=None,
        claim_count_names=None,
        claim_paid_names=None,
        claim_count_scale=None,
        claim_paid_scale=None,
        preprocess=None,
        # Network lambdas
        expected_cost_network=penalised_unexplainability_network,
        development_network=penalised_unexplainability_network,
        # Actual Parameters
        l1_l2_lin_pricing=0.01,
        l1_l2_res_pricing=0.01,
        dense_layers_pricing=3,
        dense_size_pricing=64,
        l1_l2_lin_development=0.01,
        l1_l2_res_development=0.01,
        dense_layers_development=3,
        dense_size_development=64,
        epochs=None,
        clipnorm=10000,
        l1_lin_pricing=None,
        l2_lin_pricing=None,
        l1_res_pricing=None,
        l2_res_pricing=None,
        l1_lin_development=None,
        l2_lin_development=None,
        l1_res_development=None,
        l2_res_development=None,
        **kwargs
    ):

        # Default l1/l2
        l1_lin_pricing = l1_l2_lin_pricing if l1_lin_pricing is None else l1_lin_pricing
        l2_lin_pricing = l1_l2_lin_pricing if l2_lin_pricing is None else l2_lin_pricing

        l1_res_pricing = l1_l2_res_pricing if l1_res_pricing is None else l1_res_pricing
        l2_res_pricing = l1_l2_res_pricing if l2_res_pricing is None else l2_res_pricing

        l1_lin_development = (
            l1_l2_lin_pricing if l1_lin_development is None else l1_lin_development
        )
        l2_lin_development = (
            l1_l2_lin_pricing if l2_lin_development is None else l2_lin_development
        )

        l1_res_development = (
            l1_l2_lin_pricing if l1_res_development is None else l1_res_development
        )
        l2_res_development = (
            l1_l2_lin_pricing if l2_res_development is None else l2_res_development
        )

        if dataset is not None:
            self.variate_names = dataset.variate_names
            self.category_names = dataset.category_names
            self.category_levels = dataset.category_levels

            self.X_columns = dataset.feature_names  # dataset.features.columns.tolist()
            self.claim_count_names = dataset.claim_count.columns.tolist()
            self.claim_paid_names = dataset.claim_paid.columns.tolist()

            self.preprocess = dataset.preprocess

            # Loss scale
            self.claim_count_scale = 1 / dataset.claim_count.mean().sum()
            self.claim_paid_scale = 1 / dataset.claim_paid.var().sum()

        else:
            self.variate_names = variate_names
            self.category_names = category_names
            self.category_levels = category_levels

            self.X_columns = feature_names
            self.claim_count_names = claim_count_names
            self.claim_paid_names = claim_paid_names

            self.preprocess = preprocess

            # Loss scale
            self.claim_count_scale = claim_count_scale
            self.claim_paid_scale = claim_paid_scale

        # Model parameters
        self.l1_lin_pricing = l1_lin_pricing
        self.l2_lin_pricing = l2_lin_pricing
        self.l1_res_pricing = l1_res_pricing
        self.l2_res_pricing = l2_res_pricing
        self.dense_layers_pricing = dense_layers_pricing
        self.dense_size_pricing = dense_size_pricing

        self.l1_lin_development = l1_lin_development
        self.l2_lin_development = l2_lin_development
        self.l1_res_development = l1_res_development
        self.l2_res_development = l2_res_development
        self.dense_layers_development = dense_layers_development
        self.dense_size_development = dense_size_development

        self.expected_cost_network = expected_cost_network
        self.development_network = development_network

        self.epochs = epochs
        self.clipnorm = clipnorm
        self.kwargs = kwargs

    def filter_sk_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`
        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.get_params().items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)

        return res

    def fit(
        self,
        X,
        y,
        w=None,
        optimizer=None,
        claim_count_loss="poisson",
        claim_payment_loss="mean_squared_error",
        epochs=None,
        *args,
        **kwargs
    ):
        """ A reference implementation of a fitting function.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The training input samples.
            y : array-like, shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in
                regression).
            optimizer: optimizer used by Keras
            loss: loss used by Keras
            *args, **kwargs: passed onto Keras model.fit

            Returns
            -------
            self : object
                Returns self.
        """

        # First, neural nets need data shuffling
        X, y = shuffle(X, y, random_state=0)

        X, y_new = check_X_y(X, np.nan_to_num(y), accept_sparse=True, multi_output=True)

        output_names = self.claim_count_names + self.claim_paid_names
        y_dict = {f: y for f, y in zip(output_names, list(y_new.T))}
        x_dict = {f: x for f, x in zip(self.X_columns, list(X.T))}

        dev_dim = len(self.claim_count_names)

        # Clear Keras graph between each step
        K.clear_session()

        # Input layer
        variate_inputs, categorical_inputs = network_inputs(
            variate_features=self.variate_names,
            categorical_features=self.category_names,
        )

        # Embeddings
        inputs_with_embeddings = network_embeddings(
            variate_inputs=variate_inputs,
            categorical_inputs=categorical_inputs,
            categorical_dimensions=self.category_levels,
        )

        # Risk pricing
        risk_pricing = self.expected_cost_network(
            inputs=inputs_with_embeddings,
            response=["ultimate_claim_count", "ultimate_claim_size"],
            name_prefix="risk_pricing_",
            l1_lin=self.l1_lin_pricing,
            l2_lin=self.l2_lin_pricing,
            l1_res=self.l1_res_pricing,
            l2_res=self.l2_res_pricing,
            dense_layers=self.dense_layers_pricing,
            dense_size=self.dense_size_pricing,
        )

        # Exponentiate

        # Percentage Developed
        percentage_developed = self.development_network(
            inputs=inputs_with_embeddings,
            response=self.claim_count_names + self.claim_paid_names,
            name_prefix="develop_",
            l1_lin=self.l1_lin_development,
            l2_lin=self.l2_lin_development,
            l1_res=self.l1_res_development,
            l2_res=self.l2_res_development,
            dense_layers=self.dense_layers_development,
            dense_size=self.dense_size_development,
        )

        # Softmax
        count_developed = concatenate(percentage_developed[0:dev_dim])
        count_developed = Activation("softmax")(count_developed)

        paid_developed = concatenate(percentage_developed[dev_dim:])
        paid_developed = Activation("softmax")(paid_developed)

        # Multiply it together - count
        count_outputs = [
            Lambda(lambda t: [K.exp(t[0]) * t[1]], name=nm)(
                [
                    risk_pricing[0],
                    Lambda(lambda t: t[:, i], output_shape=(1,))(count_developed),
                ]
            )
            for i, nm in enumerate(self.claim_count_names)
        ]

        # Multiply it together - Paid
        paid_outputs = [
            Lambda(lambda t: [K.exp(t[0] + t[1]) * t[2]], name=nm)(
                [
                    risk_pricing[0],
                    risk_pricing[1],
                    Lambda(lambda t: t[:, i], output_shape=(1,))(paid_developed),
                ]
            )
            for i, nm in enumerate(self.claim_paid_names)
        ]

        # Model
        self.model = Model(
            inputs=variate_inputs + categorical_inputs,
            outputs=count_outputs + paid_outputs,
        )

        # Losses
        frequency_losses = {x: claim_count_loss for x in self.claim_count_names}
        payments_losses = {x: claim_payment_loss for x in self.claim_paid_names}

        # Complete loss function
        loss = {**frequency_losses, **payments_losses}

        # Loss Weights
        frequency_loss_weights = {
            x: self.claim_count_scale for x in self.claim_count_names
        }
        payments_loss_weights = {
            x: self.claim_paid_scale for x in self.claim_paid_names
        }

        loss_weights = {**frequency_loss_weights, **payments_loss_weights}

        # Sample weight - if none, use (ones otherwise weight) * exclude NaN
        if w is None:
            w = np.ones(X.shape[0])

        sample_weight = {
            f: np.where(np.isnan(y), 0.0, w) for f, y in zip(output_names, list(y.T))
        }

        if optimizer is None:
            optimizer = optimizers.Adam(lr=0.01, clipnorm=self.clipnorm)

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

        if epochs is not None:
            self.model.fit(
                x=x_dict,
                y=y_dict,
                epochs=epochs,
                sample_weight=sample_weight,
                *args,
                **kwargs
            )
        else:
            if self.epochs is not None:
                self.model.fit(
                    x=x_dict,
                    y=y_dict,
                    sample_weight=sample_weight,
                    epochs=self.epochs,
                    *args,
                    **kwargs
                )
            else:
                self.model.fit(
                    x=x_dict, y=y_dict, sample_weight=sample_weight, *args, **kwargs
                )

        # Finished fitting!
        self.is_fitted_ = True

        # Store training predictions at various points in network
        # for explaining relationships
        # self.train_components = self.predict_components(X)

        return self

    def get_params(self, deep=True):
        # return params

        return {
            "epochs": self.epochs,
            "variate_names": self.variate_names,
            "category_names": self.category_names,
            "category_levels": self.category_levels,
            "feature_names": self.X_columns,
            "claim_count_names": self.claim_count_names,
            "claim_paid_names": self.claim_paid_names,
            "preprocess": self.preprocess,
            "claim_count_scale": self.claim_count_scale,
            "claim_paid_scale": self.claim_paid_scale,
            "l1_l2_lin_pricing": self.l1_lin_pricing,
            "l1_lin_pricing": self.l1_lin_pricing,
            "l2_lin_pricing": self.l2_lin_pricing,
            "l1_l2_res_pricing": self.l1_res_pricing,
            "l1_res_pricing": self.l1_res_pricing,
            "l2_res_pricing": self.l2_res_pricing,
            "dense_layers_pricing": self.dense_layers_pricing,
            "dense_size_pricing": self.dense_size_pricing,
            "l1_l2_lin_development": self.l1_lin_development,
            "l1_lin_development": self.l1_lin_development,
            "l2_lin_development": self.l2_lin_development,
            "l1_l2_res_development": self.l1_res_development,
            "l1_res_development": self.l1_res_development,
            "l2_res_development": self.l2_res_development,
            "dense_layers_development": self.dense_layers_development,
            "dense_size_development": self.dense_size_development,
            "clipnorm": self.clipnorm,
        }

    def get_weights(self):
        """ Returns weights as a dictionary """
        names = [
            layer.name + "/" + weight.name.split("/")[1]
            for layer in self.model.layers
            for weight in layer.weights
        ]
        weights = self.model.get_weights()
        weights_dict = {a: b for a, b in zip(names, weights)}
        return weights_dict

    def linear_vs_residual(self):
        """ Compare linear effect vs residual effect """

        weights_dict = self.get_weights()

        claim_count_ensembles = [
            "develop_{}/kernel:0".format(x) for x in self.claim_count_names
        ]
        claim_paid_ensembles = [
            "develop_{}/kernel:0".format(x) for x in self.claim_paid_names
        ]

        names_list = (
            [
                "risk_pricing_ultimate_claim_count/kernel:0",
                "risk_pricing_ultimate_claim_size/kernel:0",
            ]
            + claim_count_ensembles
            + claim_paid_ensembles
        )

        weights = np.hstack([weights_dict[x] for x in names_list])

        return pd.DataFrame(
            weights.T,
            columns=["Linear", "Residual Network"],
            index=(
                [
                    "risk_pricing_ultimate_claim_count",
                    "risk_pricing_ultimate_claim_size",
                ]
                + self.claim_count_names
                + self.claim_paid_names
            ),
        )

    def plot_graph(self):
        """ Plots the model graph to graphviz
        """
        return SVG(model_to_dot(self.model).create(prog="dot", format="svg"))

    def plot_pdp_1d(self, dataset, feature, response):
        """ Explains response vs feature """

        # Use transformed df if categorical, otherwise
        # use the untransformed dataset with a transformer
        if hasattr(dataset.features[feature], "cat"):
            # Transformed values
            df = pd.DataFrame(
                self.preprocess.transform(dataset.features.values),
                columns=dataset.feature_names,
            )

            def data_transformer(df):
                return df

        else:
            df = dataset.features

            def data_transformer(df):
                return pd.DataFrame(
                    self.preprocess.transform(df.values), columns=dataset.feature_names
                )

        pdp_iso = pdp.pdp_isolate(
            model=self,
            dataset=df,
            predict_kwds={"response_variable": response},
            model_features=dataset.feature_names,
            data_transformer=data_transformer,
            feature=feature,
        )

        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_iso, feature_name=feature)

        if hasattr(dataset.features[feature], "cat"):
            category_index = self.category_names.index(feature)
            category_labels = (
                self.preprocess.named_transformers_["ordinalencoder"]
                .categories_[category_index]
                .tolist()
            )
            axes["pdp_ax"].set_xticklabels(category_labels)

        return fig, axes

    def plot_pdp_frequency(self, dataset, feature):
        """ Explains frequency vs feature"""
        return self.plot_pdp_1d(dataset, feature, "frequency")

    def plot_pdp_size(self, dataset, feature):
        """ Explains severity vs feature"""
        return self.plot_pdp_1d(dataset, feature, "size")

    def ppci(self, dataset):
        """ Tabulates a triangle of PPCI and the selections
        """
        pred = self.predict(dataset.X())
        pred[dataset.origin.name] = dataset.origin.values
        pred_sum = pred.groupby([dataset.origin.name]).agg("sum")

        # Count predictions
        count_df = dataset.claim_count.copy()
        count_df[dataset.origin.name] = dataset.origin.values

        display(count_df.groupby([dataset.origin.name]).agg("sum"))
        display(pred_sum[self.claim_count_names])

        # Paid predictions
        paid_df = dataset.claim_paid.copy()
        paid_df[dataset.origin.name] = dataset.origin.values

        display(paid_df.groupby([dataset.origin.name]).agg("sum"))
        display(pred_sum[self.claim_paid_names])

    def predict(self, X, response_variable=None, *args, **kwargs):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        response_variable: default None, EXP_COST or PERC_DEV
            The prediction to return

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        if response_variable is None:
            X = check_array(X, accept_sparse=True)
            check_is_fitted(self, "is_fitted_")
            x_dict = {f: x for f, x in zip(self.X_columns, list(X.T))}

            # Need to set batch_size=1 to avoid errors in the current model
            # structure. This should be reviewed in future enhancements
            predictions = self.model.predict(x_dict, batch_size=1, *args, **kwargs)

            return pd.DataFrame(
                np.hstack(predictions),
                columns=self.claim_count_names + self.claim_paid_names,
            )

        elif response_variable == "frequency":
            components = self.predict_components(X, *args, **kwargs)
            return np.exp(components["risk_pricing_ultimate_claim_count"])

        elif response_variable == "size":
            components = self.predict_components(X, *args, **kwargs)
            return np.exp(components["risk_pricing_ultimate_claim_size"])

    def predict_components(self, X, *args, **kwargs):
        """ Predicts expected cost and percentage developed

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        predictions: dict of exp_cost (expected cost), perc_dev
            (percentage developed)
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        input_names = [i.name for i in self.model.inputs]
        output_names = (
            self.claim_count_names
            + self.claim_paid_names
            + ["risk_pricing_ultimate_claim_count", "risk_pricing_ultimate_claim_size"]
        )

        layers = [l for l in self.model.layers if l.name in output_names]
        output_names_ordered = [l.name for l in layers]

        intermediate_model = Model(
            inputs=self.model.inputs, outputs=[l.output for l in layers]
        )

        x_dict = {f: x for f, x in zip(self.X_columns, list(X.T))}

        predictions = intermediate_model.predict(x_dict, batch_size=1)

        return {n: p for n, p in zip(output_names_ordered, predictions)}

    def print_layers(self):
        """ Print layers and shapes """
        for layer in self.model.layers:
            print(layer.name)
            print(layer.output_shape)

    def score(self, X, y, **kwargs):
        """Returns the mean loss on the given test data and labels.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.
        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.
        """

        X, y_new = check_X_y(X, np.nan_to_num(y), accept_sparse=True, multi_output=True)

        output_names = self.claim_count_names + self.claim_paid_names

        y_dict = {f: y for f, y in zip(output_names, list(y_new.T))}
        x_dict = {f: x for f, x in zip(self.X_columns, list(X.T))}

        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
        loss = self.model.evaluate(x_dict, y_dict, **kwargs)

        if isinstance(loss, list):
            # If NaN just make it very poor.
            if np.isnan(loss[0]):
                return np.nan_to_num(np.NINF)
            else:
                return -loss[0]
        else:
            if np.isnan(loss):
                return np.nan_to_num(np.NINF)
            else:
                return -loss
