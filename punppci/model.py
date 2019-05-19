from .networks import (
    network_inputs,
    penalised_unexplainability_network,
    network_embeddings,
    NonNegativeWeightedAverage,
)

from tempfile import SpooledTemporaryFile, TemporaryFile

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle

from keras.layers import Lambda, Reshape, Input
from keras.models import load_model, Model, Sequential
from keras.utils.vis_utils import model_to_dot
from keras.utils.generic_utils import has_arg
from keras.layers import concatenate, Activation, Multiply
from keras.callbacks import Callback
from keras import backend as K
from keras import optimizers
from keras.initializers import Constant

from IPython.display import display, SVG

from pdpbox import info_plots, pdp

import h5py
import pandas as pd
import numpy as np
import gc


class BalanceLosses(Callback):
    """ Callback that rebalances losses to the same ratio
    """

    def __init__(
        self,
        initial_count_weight,
        initial_paid_weight,
        count_loss_list,
        paid_loss_list,
        learn_rate,
    ):
        self.count_weight = initial_count_weight
        self.paid_weight = initial_paid_weight
        self.count_loss_list = count_loss_list
        self.paid_loss_list = paid_loss_list
        self.learn_rate = learn_rate

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        weights_df = pd.DataFrame.from_records([logs])

        if False:  # Remove for testing
            count_loss = weights_df[self.count_loss_list].sum(axis=1).values[0]
            paid_loss = weights_df[self.paid_loss_list].sum(axis=1).values[0]
            if count_loss > 0 and paid_loss > 0:
                new_count_loss_weight = 1 / count_loss
                new_paid_loss_weight = 1 / paid_loss

                current_loss = weights_df["loss"][0]

                K.set_value(
                    self.count_weight, current_loss * 0.5 / new_count_loss_weight
                )
                K.set_value(self.paid_weight, current_loss * 0.5 / new_paid_loss_weight)

                # Reset learn rate
                K.set_value(self.model.optimizer.lr, 0.001)

                print(
                    "Loss at epoch 2 is {}: Reset loss weights to {} for count and {} for paid".format(
                        current_loss, new_count_loss_weight, new_paid_loss_weight
                    )
                )
            else:
                print(
                    "Loss at epoch 2 is: Count loss {}, Paid loss {} - maintain current loss weights.".format(
                        count_loss, paid_loss
                    )
                )


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
        claim_count_initializer=None,
        claim_size_initializer=None,
        develop_count_initializer=None,
        develop_paid_initializer=None,
        preprocess=None,
        # Network lambdas
        expected_cost_network=penalised_unexplainability_network,
        development_network=penalised_unexplainability_network,
        # Actual Parameters
        l1_l2_lin_pricing=0.001,
        l1_l2_res_pricing=0.001,
        dense_layers_pricing=3,
        dense_size_pricing=64,
        l1_l2_lin_development=0.01,
        l1_l2_res_development=0.01,
        dense_layers_development=3,
        dense_size_development=64,
        epochs=20,
        clipnorm=None,
        l1_lin_pricing=None,
        l2_lin_pricing=None,
        l1_res_pricing=None,
        l2_res_pricing=None,
        l1_lin_development=None,
        l2_lin_development=None,
        l1_res_development=None,
        l2_res_development=None,
        learn_rate=0.01,
        **kwargs,
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
            self.claim_paid_scale = 1 / dataset.claim_paid.mean().sum()

            # Initilisers
            count_ultimates = dataset.chain_ladder_count("ultimates")
            ppci_selections = dataset.ppci("selections")

            log_frequency_average = [
                np.log(dataset.claim_count[x].sum() + K.epsilon())
                - np.log(np.where(dataset.claim_count[x].isna(), 0, dataset.w()).sum())
                + K.epsilon()
                for x in dataset.claim_count
            ]

            self.claim_count_initializer = np.log(
                count_ultimates.sum() / dataset.w().sum()
            )
            self.claim_size_initializer = np.log(ppci_selections.sum())

            self.develop_count_initializer = log_frequency_average
            self.develop_paid_initializer = np.log(
                ppci_selections.values + K.epsilon()
            ).tolist()

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

            self.claim_count_initializer = claim_count_initializer
            self.claim_size_initializer = claim_size_initializer
            self.develop_count_initializer = develop_count_initializer
            self.develop_paid_initializer = develop_paid_initializer

        assert len(self.claim_count_names) == len(
            self.claim_paid_names
        ), "Claim count should be same dimensions as claim paid!"

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
        self.learn_rate = learn_rate
        self.kwargs = kwargs

    def x_dict(self, X, y=None, y_dim=None):
        predictors = {f: x for f, x in zip(self.X_columns, list(X.T))}

        if y is None:
            weights_mask_count = np.ones((X.shape[0], y_dim))
            weights_mask_paid = np.ones((X.shape[0], y_dim))
        else:
            weights_mask_count = np.where(np.isnan(y[:, 0:y_dim]), 0, 1)
            weights_mask_paid = np.where(np.isnan(y[:, y_dim:]), 0, 1)

        x_dict = {
            "weights_mask_count": weights_mask_count,
            "weights_mask_paid": weights_mask_paid,
            **predictors,
        }
        return x_dict

    def y_dict(self, y):
        return {
            "claim_count": y[:, 0 : len(self.claim_count_names)],
            "claim_paid": y[:, len(self.claim_count_names) :],
        }

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
        **kwargs,
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
        y_dict = self.y_dict(y_new)
        x_dict = self.x_dict(X, y, len(self.claim_count_names))

        dev_dim = len(self.claim_count_names)

        # Clear Keras graph between each step
        K.clear_session()

        # Input layer
        variate_inputs, categorical_inputs = network_inputs(
            variate_features=["origin"] + self.variate_names,
            categorical_features=self.category_names,
        )

        weights_mask_count = Input(
            shape=(len(self.claim_count_names),), name="weights_mask_count"
        )
        weights_mask_paid = Input(
            shape=(len(self.claim_paid_names),), name="weights_mask_paid"
        )

        # Embeddings
        embeddings_count = network_embeddings(
            variate_inputs=variate_inputs,
            categorical_inputs=categorical_inputs,
            categorical_dimensions=self.category_levels,
            name_prefix="embed_count",
        )

        embeddings_size = network_embeddings(
            variate_inputs=variate_inputs,
            categorical_inputs=categorical_inputs,
            categorical_dimensions=self.category_levels,
            name_prefix="embed_size",
        )

        embeddings_develop_count = network_embeddings(
            variate_inputs=variate_inputs,
            categorical_inputs=categorical_inputs,
            categorical_dimensions=self.category_levels,
            name_prefix="embed_develop_count",
        )

        embeddings_develop_paid = network_embeddings(
            variate_inputs=variate_inputs,
            categorical_inputs=categorical_inputs,
            categorical_dimensions=self.category_levels,
            name_prefix="embed_develop_paid",
        )

        # Risk pricing
        risk_count = self.expected_cost_network(
            inputs=embeddings_count,
            response=["ultimate_claim_count"],
            name_prefix="risk_count_",
            l1_lin=self.l1_lin_pricing,
            l2_lin=self.l2_lin_pricing,
            l1_res=self.l1_res_pricing,
            l2_res=self.l2_res_pricing,
            dense_layers=self.dense_layers_pricing,
            dense_size=self.dense_size_pricing,
            bias_initializer=Constant(self.claim_count_initializer),
            activation="exponential",
        )

        risk_size = self.expected_cost_network(
            inputs=embeddings_size,
            response=["ultimate_claim_size"],
            name_prefix="risk_size_",
            l1_lin=self.l1_lin_pricing,
            l2_lin=self.l2_lin_pricing,
            l1_res=self.l1_res_pricing,
            l2_res=self.l2_res_pricing,
            dense_layers=self.dense_layers_pricing,
            dense_size=self.dense_size_pricing,
            bias_initializer=Constant(self.claim_size_initializer),
            activation="exponential",
        )

        # Exponentiate

        # Percentage Developed
        developed_count = self.development_network(
            inputs=embeddings_develop_count,
            response=["claim_count"],
            name_prefix="develop_count_",
            l1_lin=self.l1_lin_development,
            l2_lin=self.l2_lin_development,
            l1_res=self.l1_res_development,
            l2_res=self.l2_res_development,
            dense_layers=self.dense_layers_development,
            dense_size=self.dense_size_development,
            n_dim=len(self.claim_count_names),
            bias_initializer=Constant(self.develop_count_initializer),
        )

        developed_paid = self.development_network(
            inputs=embeddings_develop_paid,
            response=["claim_paid"],
            name_prefix="develop_paid_",
            l1_lin=self.l1_lin_development,
            l2_lin=self.l2_lin_development,
            l1_res=self.l1_res_development,
            l2_res=self.l2_res_development,
            dense_layers=self.dense_layers_development,
            dense_size=self.dense_size_development,
            n_dim=len(self.claim_count_names),
            bias_initializer=Constant(self.develop_paid_initializer),
        )

        # Softmax
        count_developed = Activation("softmax")(developed_count[0])

        paid_developed = Activation("softmax")(developed_paid[0])

        # Multiply it together - count
        # Need to have i as an explicit argument to enable serialisation
        # Refer https://github.com/keras-team/keras/issues/4875#issuecomment-455496677

        count_outputs = [
            Multiply(name="claim_count")(
                [risk_count[0], count_developed, weights_mask_count]
            )
        ]

        paid_outputs = [
            Multiply(name="claim_paid")(
                [risk_count[0], risk_size[0], paid_developed, weights_mask_paid]
            )
        ]

        # Model
        self.model = Model(
            inputs=[weights_mask_count, weights_mask_paid]
            + variate_inputs
            + categorical_inputs,
            outputs=count_outputs + paid_outputs,
        )

        # Losses
        frequency_losses = {"claim_count": claim_count_loss}
        payments_losses = {"claim_paid": claim_payment_loss}

        # Complete loss function
        loss = {**frequency_losses, **payments_losses}

        # Loss Weights
        count_loss_weight = K.variable(self.claim_count_scale)
        paid_loss_weight = K.variable(self.claim_paid_scale)

        # For loss rebalancing
        print(
            "Set initial loss weights to be {} for count and {} for paid".format(
                self.claim_count_scale, self.claim_paid_scale
            )
        )

        loss_weights = self.make_loss_weights(count_loss_weight, paid_loss_weight)

        count_loss_list = ["claim_count_loss"]
        paid_loss_list = ["claim_paid_loss"]

        callback = BalanceLosses(
            initial_count_weight=count_loss_weight,
            initial_paid_weight=paid_loss_weight,
            count_loss_list=count_loss_list,
            paid_loss_list=paid_loss_list,
            learn_rate=self.learn_rate,
        )

        # Sample weight - if none, use (ones otherwise weight) * exclude NaN
        # The records at the right end of the triangle being mostly zero may
        # be a cause of the exploding gradients. Consequently instead of 0
        # set to K.epsilon * 2

        if w is None:
            w = np.ones(X.shape[0])

        sample_weight = {"claim_count": w, "claim_paid": w}

        if optimizer is None:
            if self.clipnorm is None:
                optimizer = optimizers.Adam(lr=self.learn_rate)
            else:
                optimizer = optimizers.Adam(lr=self.learn_rate, clipnorm=self.clipnorm)

        # Compile
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

        print(self.model.summary())

        if epochs is not None:
            self.model.fit(
                x=x_dict,
                y=y_dict,
                epochs=epochs,
                sample_weight=sample_weight,
                callbacks=[callback],
                *args,
                **kwargs,
            )
        else:
            if self.epochs is not None:
                self.model.fit(
                    x=x_dict,
                    y=y_dict,
                    sample_weight=sample_weight,
                    callbacks=[callback],
                    epochs=self.epochs,
                    *args,
                    **kwargs,
                )
            else:
                self.model.fit(
                    x=x_dict,
                    y=y_dict,
                    sample_weight=sample_weight,
                    callbacks=[callback],
                    *args,
                    **kwargs,
                )

        # Finished fitting!
        self.is_fitted_ = True

        # Store training predictions at various points in network
        # for explaining relationships
        # self.train_components = self.predict_components(X)

        # Reset loss weights for consistent scoring
        # and https://github.com/keras-team/keras/issues/9444

        loss_weights = self.make_loss_weights(
            self.claim_count_scale, self.claim_paid_scale
        )
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

        # Store network as a binary object
        # K.clear_session() is needed to conserve memory, but wipes all networks
        # so storing allows maintaining more than one of this regressor.
        with SpooledTemporaryFile() as tmpf:
            h5f = h5py.File(tmpf)
            self.model.save(h5f)
            h5f.close()
            del h5f
            tmpf.seek(0)
            self.model_binary = tmpf.read()
            tmpf.close()

        gc.collect()
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

        names_list = [
            "risk_count_blend_weight_ultimate_claim_count/kernel:0",
            "risk_size_blend_weight_ultimate_claim_size/kernel:0",
            "develop_count_blend_weight_claim_count/kernel:0",
            "develop_paid_blend_weight_claim_paid/kernel:0",
        ]

        weights = np.vstack([weights_dict[x] for x in names_list])

        return pd.DataFrame(
            weights,
            columns=["Linear", "Residual Network"],
            index=(
                [
                    "risk_count_ultimate_claim_count",
                    "risk_size_ultimate_claim_size",
                    "development_count",
                    "development_paid",
                ]
            ),
        )

    def make_loss_weights(self, claim_count_scale, claim_paid_scale):
        # Loss Weights
        frequency_loss_weights = {"claim_count": claim_count_scale}
        payments_loss_weights = {"claim_paid": claim_paid_scale}

        loss_weights = {**frequency_loss_weights, **payments_loss_weights}

        return loss_weights

    def plot_graph(self):
        """ Plots the model graph to graphviz
            This version omits the final multiplication and split
            For simplicity
        """
        input_names = [i.name for i in self.model.inputs]
        output_names = (
            # self.claim_count_names
            # + self.claim_paid_names
            [
                "risk_count_blend_weight_ultimate_claim_count",
                "risk_size_blend_weight_ultimate_claim_size",
                "develop_count_blend_weight_claim_count",
                "develop_paid_blend_weight_claim_paid",
            ]
        )

        layers = [l for l in self.model.layers if l.name in output_names]
        output_names_ordered = [l.name for l in layers]

        intermediate_model = Model(
            inputs=self.model.inputs, outputs=[l.output for l in layers]
        )

        return SVG(model_to_dot(intermediate_model).create(prog="dot", format="svg"))

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
            x_dict = self.x_dict(X, None, len(self.claim_count_names))

            # Reload model from model binary - optimizer may have cleared it
            self.reload_model()

            predictions = self.model.predict(x_dict, *args, **kwargs)

            return pd.DataFrame(
                np.hstack(predictions),
                columns=self.claim_count_names + self.claim_paid_names,
            )

        elif response_variable == "frequency":
            components = self.predict_components(X, *args, **kwargs)
            gc.collect()
            return np.exp(components["risk_count_blend_weight_ultimate_claim_count"])

        elif response_variable == "size":
            components = self.predict_components(X, *args, **kwargs)
            gc.collect()
            return np.exp(components["risk_size_blend_weight_ultimate_claim_size"])

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

        # Reload model from model binary - optimizer may have cleared it
        self.reload_model()

        input_names = [i.name for i in self.model.inputs]
        output_names = [
            "risk_count_blend_weight_ultimate_claim_count",
            "risk_size_blend_weight_ultimate_claim_size",
        ]

        layers = [l for l in self.model.layers if l.name in output_names]
        output_names_ordered = [l.name for l in layers]

        intermediate_model = Model(
            inputs=self.model.inputs, outputs=[l.output for l in layers]
        )

        x_dict = self.x_dict(X, None, len(self.claim_count_names))

        predictions = intermediate_model.predict(x_dict)

        return {n: p for n, p in zip(output_names_ordered, predictions)}

    def print_layers(self):
        """ Print layers and shapes """
        for layer in self.model.layers:
            print(layer.name)
            print(layer.output_shape)

    def reload_model(self):
        """ Reload model from stored binary """
        with TemporaryFile() as tmpf:
            tmpf.write(self.model_binary)
            tmpf.seek(0)
            h5f = h5py.File(tmpf)
            self.model = load_model(
                h5f,
                custom_objects={
                    "NonNegativeWeightedAverage": NonNegativeWeightedAverage
                },
            )
            h5f.close()
            tmpf.close()
            del h5f

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

        # Reload model from model binary - optimizer may have cleared it
        self.reload_model()

        output_names = ["claim_count", "claim_paid"]

        y_dict = self.y_dict(y_new)
        x_dict = self.x_dict(X, None, len(self.claim_count_names))

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
