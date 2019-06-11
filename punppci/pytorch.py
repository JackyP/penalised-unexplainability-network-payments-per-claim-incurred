import torch
from torch import nn
import torch.nn.functional as F
from warnings import warn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, LRScheduler, EarlyStopping, GradientNormClipping
from skorch.callbacks.lr_scheduler import CyclicLR
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from skopt import BayesSearchCV
from pdpbox import info_plots, pdp
import gc

# To do list:
#  - Documentation
#  - Framework
#  - y normalisation
EPSILON = 0.00001


class PUNPPCIClaimModule(nn.Module):
    def __init__(
        self,
        feature_dim=None,
        output_dim=None,
        y_mean=None,
        nonlin=F.selu,
        # l1_l2_lin_pricing=0.001,
        # l1_l2_res_pricing=0.001,
        # dense_layers_pricing=3,
        layer_size=100,
    ):
        super(PUNPPCIClaimModule, self).__init__()

        # Initialise
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        # self.l1_l2_lin_pricing = l1_l2_lin_pricing
        # self.l1_l2_res_pricing = l1_l2_res_pricing

        # Pricing - Neural Network
        self.non_lin_0 = nonlin
        self.non_lin_1 = nonlin
        self.non_lin_2 = nonlin

        self.dense_pricing_0 = nn.Linear(feature_dim, layer_size)
        self.dense_pricing_1 = nn.Linear(layer_size, layer_size)
        self.dense_pricing_2 = nn.Linear(layer_size, layer_size)

        # Pricing - Neural Network Output
        self.dropout = nn.Dropout(0.5)
        self.count_residual_0 = nn.Linear(layer_size, 1)
        self.paid_residual_0 = nn.Linear(layer_size, 1)

        # Pricing - Linear
        self.count_linear_0 = nn.Linear(feature_dim, 1)
        self.paid_linear_0 = nn.Linear(feature_dim, 1)

        # Pricing - Blend
        self.count_blend_w = nn.Parameter(torch.Tensor(1))
        self.paid_blend_w = nn.Parameter(torch.Tensor(1))

        # Pricing - Spread
        self.count_spread = nn.Parameter(torch.Tensor(output_dim))
        self.paid_spread = nn.Parameter(torch.Tensor(output_dim))

        self.count_residual_spread = nn.Linear(layer_size, output_dim)
        self.paid_residual_spread = nn.Linear(layer_size, output_dim)

        # GLM components - Initialise at zero
        nn.init.zeros_(self.count_linear_0.weight)
        nn.init.zeros_(self.paid_linear_0.weight)

        if y_mean is None:
            nn.init.normal(self.count_spread.data, mean=0, std=0.1)
            nn.init.normal(self.paid_spread.data, mean=0, std=0.1)
        else:
            # print(y_mean)

            freq_mean = y_mean[0:output_dim] + EPSILON
            paid_mean = y_mean[output_dim : 2 * output_dim] + EPSILON

            self.count_spread.data = torch.Tensor(np.log(freq_mean))

            # self.count_linear_0.bias = nn.Parameter(torch.from_numpy(np.array(np.log(freq_mean).sum())))
            # self.count_residual_0.bias = nn.Parameter(torch.from_numpy(np.array(np.log(freq_mean).sum())))

            self.paid_spread.data = torch.Tensor(np.log(paid_mean))

            # self.paid_linear_0.bias = nn.Parameter(torch.from_numpy(np.array(np.log(paid_mean).sum())))
            # self.paid_residual_0.bias = nn.Parameter(torch.from_numpy(np.array(np.log(paid_mean).sum())))

    def forward(self, X, **kwargs):

        # Split
        X, wc, wp = list(
            torch.split(X, [self.feature_dim, self.output_dim, self.output_dim], dim=1)
        )

        # Resnet
        Xr = self.dropout(X)
        Xr = self.non_lin_0(self.dense_pricing_0(Xr))
        Xr = self.non_lin_1(self.dense_pricing_1(Xr))
        Xr = self.non_lin_2(self.dense_pricing_2(Xr))

        count_residual = self.count_residual_0(Xr)
        paid_residual = self.paid_residual_0(Xr)

        # Linear
        count_linear = self.count_linear_0(X)
        paid_linear = self.paid_linear_0(X)

        # Spread - Constant
        count_s = self.count_spread
        paid_s = self.paid_spread

        count_dev_linear = wc * torch.exp(count_linear + F.log_softmax(count_s))
        paid_dev_linear = wp * torch.exp(paid_linear + F.log_softmax(paid_s))

        # Spread - Residual
        count_sr = self.count_residual_spread(Xr)
        paid_sr = self.paid_residual_spread(Xr)

        count_dev_residual = wc * torch.exp(
            count_linear + count_residual + F.log_softmax(count_s + count_sr)
        )
        paid_dev_residual = wp * torch.exp(
            paid_linear + paid_residual + F.log_softmax(paid_s + paid_sr)
        )

        # Blended
        count_blend_weight = torch.exp(self.count_blend_w) / (
            1 + torch.exp(self.count_blend_w)
        )
        paid_blend_weight = torch.exp(self.paid_blend_w) / (
            1 + torch.exp(self.paid_blend_w)
        )

        count_dev_blended = (
            count_dev_residual * count_blend_weight
            + count_dev_linear * (1 - count_blend_weight)
        )

        paid_dev_blended = paid_dev_residual * paid_blend_weight + paid_dev_linear * (
            1 - paid_blend_weight
        )

        return torch.cat(
            [
                count_dev_blended,
                paid_dev_blended,
                count_dev_linear,
                paid_dev_linear,
                count_dev_residual,
                paid_dev_residual,
            ],
            dim=1,
        )


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
            else:
                print(n)
                print(p)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")


class CheckNaN(Callback):
    def __init__(self):
        self.params = None

    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_begin(self, net, **kwargs):
        self.params = net.module.named_parameters()

    def on_epoch_end(self, net, **kwargs):

        # max_grad = 0
        # params = net.module.named_parameters()
        # for n, p in params:
        #     max_grad = np.maximum(max_grad, p.grad.abs().max())
        # print("Max grad: {}".format(max_grad))

        if self.critical_epoch_ > -1:
            return
        # look at the validation accuracy of the last epoch

        if np.isnan(net.history[-1, "train_loss"]):

            self.critical_epoch_ = len(net.history)
            print("Maximum Gradients and Weights")
            # params = net.module.named_parameters()
            if self.params is not None:
                for n, p in self.params:
                    print("Gradients {}: {}".format(n, p.grad.abs().max()))
                    print("Weights {}: {}".format(n, p.data.abs().max()))

        plot_grad_flow(net.module.named_parameters())


class CheckMean(Callback):
    def __init__(self, X, num_cols, modulo):
        self.num_cols = num_cols
        self.modulo = modulo
        self.X = X

    def on_epoch_end(self, net, **kwargs):
        # Test every 10 epochs
        if net.history[-1]["epoch"] % (self.modulo + 1) == 0:
            avg = net._predict(self.X).mean(axis=0)
            params = net.module.named_parameters()

            num_cols = self.num_cols

            # Count and Paid Means
            print(
                "Blended  Count {}, Paid {}".format(
                    avg[0:num_cols].sum(), avg[num_cols : 2 * num_cols].sum()
                )
            )
            print(
                "Linear   Count {}, Paid {}".format(
                    avg[2 * num_cols : 3 * num_cols].sum(),
                    avg[3 * num_cols : 4 * num_cols].sum(),
                )
            )
            print(
                "Residual Count {}, Paid {}".format(
                    avg[4 * num_cols : 5 * num_cols].sum(),
                    avg[5 * num_cols : 6 * num_cols].sum(),
                )
            )

            # Spread Gradients
            for n, p in params:
                if n in ["count_spread", "count_spread", "paid_spread", "paid_spread"]:
                    print("{} data: {}".format(n, p.data))
                    print("{} grad: {}".format(n, p.grad))


class PUNPPCIClaimRegressor(BaseEstimator, NeuralNetRegressor):
    def __init__(
        self,
        claim_count_names,
        claim_paid_names,
        feature_dimension=None,
        output_dimension=None,
        # Calibrate these parameters:
        layer_size=100,
        l1_l2_linear=0.01,
        l2_residual=0.1,
        l2_bias=0.00,
        # Leave these parameters alone generally:
        batch_size=5000,
        optimizer=torch.optim.Adam,
        max_epochs=100,
        lr_range=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
        patience=5,
        clipnorm=0.1,
        input_X_datasets=["origin"],
        input_y_datasets=["claim_count", "claim_paid"],
    ):

        self.input_X_datasets = input_X_datasets
        self.input_y_datasets = input_y_datasets
        self.origin_transformer = None
        self.categorical_transformer = None
        self.variate_transformer = None

        self.optimizer = optimizer

        self.feature_dimension = feature_dimension
        self.output_dimension = output_dimension
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.layer_size = layer_size

        self.lr_range = lr_range
        # self.lr_bias = lr_bias
        # self.momentum = momentum
        self.l1_l2_linear = l1_l2_linear
        self.l2_residual = l2_residual
        self.l2_bias = l2_bias
        self.patience = patience
        self.clipnorm = clipnorm

        self.claim_count_names = claim_count_names
        self.claim_paid_names = claim_paid_names

        # For the BayesSearchCV
        self.initialized_ = True

    def fit(self, X, y, *args, w=None, **kwargs):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        X = (X - self.X_mean) / self.X_std

        if w is None:
            w = np.where(np.isnan(y), 0.0, 1.0)
        else:
            w = w * np.where(np.isnan(y), 0.0, 1.0)

        X = np.hstack([X, w]).astype(np.float32)

        y_mean = np.nanmean(y, axis=0)

        y = np.hstack([y, y, y])
        y = np.where(np.isnan(y), 0, np.maximum(EPSILON, y)).astype(np.float32)

        earlystop = EarlyStopping(patience=self.patience, threshold=0.0)
        gradclip = GradientNormClipping(gradient_clip_value=self.clipnorm)

        if X.shape[0] < self.batch_size:
            warn("Data size is small, outcomes may be odd.")
            batch_size = 128
        else:
            batch_size = self.batch_size

        # One cycle policy (with Adam)

        # Step 1: LR Range Finder
        # Test which values fit
        # Use earlystop to get an idea of epochs to 1 cycle policy as well.

        for lr in self.lr_range:

            super(PUNPPCIClaimRegressor, self).__init__(
                PUNPPCIClaimModule(
                    feature_dim=self.feature_dimension,
                    output_dim=self.output_dimension,
                    y_mean=y_mean,
                    layer_size=self.layer_size,
                ),
                *args,
                **kwargs,
                max_epochs=self.max_epochs,
                lr=lr,
                optimizer=self.optimizer,
                # optimizer__momentum=self.momentum,
                optimizer__param_groups=[
                    ("dense_pricing*", {"weight_decay": self.l2_residual}),
                    ("count_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                    ("paid_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                    ("count_linear_0.bias", {"weight_decay": self.l2_bias}),
                    ("paid_linear_0.bias", {"weight_decay": self.l2_bias}),
                    ("count_residual_0.bias", {"weight_decay": self.l2_bias}),
                    ("paid_residual_0.bias", {"weight_decay": self.l2_bias}),
                ],
                batch_size=batch_size,
                criterion=nn.MSELoss,
                callbacks=[gradclip, earlystop],
                verbose=0
            )

            self.initialize_module()

            super(PUNPPCIClaimRegressor, self).fit(X, y)

            if not np.isnan(self.history[-1]["valid_loss"]):
                self.lr_min = self.lr_range[-1]
                self.lr_max = lr
                break

        # Still broke?
        if np.isnan(self.history[-1]["valid_loss"]):
            warn(
                "This model may fail to converge on the data. Please review data and parameters."
            )
            self.lr_min = self.lr_range[-1]
            self.lr_max = 0.001

        print("Setting maximum learn rate to {}.".format(self.lr_max))

        # Step 2: Cyclic LR with expected epoch count...
        valid_losses = [x["valid_loss"] for x in self.history]
        expected_epoch_count = valid_losses.index(min(valid_losses)) + 1
        expected_epoch_count = int(np.ceil(expected_epoch_count / 2) * 2)

        print("Setting epochs for training model to {}".format(expected_epoch_count))
        cyclic_lr = LRScheduler(
            policy=CyclicLR,
            base_lr=self.lr_min,
            max_lr=self.lr_max,
            step_size_up=expected_epoch_count / 2,
            step_size_down=expected_epoch_count / 2,
        )

        # ... but still keep training for as many epochs as required.

        super(PUNPPCIClaimRegressor, self).__init__(
            PUNPPCIClaimModule(
                feature_dim=self.feature_dimension,
                output_dim=self.output_dimension,
                y_mean=y_mean,
                layer_size=self.layer_size,
            ),
            max_epochs=expected_epoch_count,
            lr=self.lr_min,
            optimizer=self.optimizer,
            # optimizer__momentum=self.momentum,
            optimizer__param_groups=[
                ("dense_pricing*", {"weight_decay": self.l2_residual}),
                ("count_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                ("paid_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                ("count_linear_0.bias", {"weight_decay": self.l2_bias}),
                ("paid_linear_0.bias", {"weight_decay": self.l2_bias}),
                ("count_residual_0.bias", {"weight_decay": self.l2_bias}),
                ("paid_residual_0.bias", {"weight_decay": self.l2_bias}),
            ],
            batch_size=batch_size,
            criterion=nn.MSELoss,
            callbacks=[
                CheckNaN(),
                CheckMean(X, self.output_dimension, expected_epoch_count),
                cyclic_lr,
                gradclip,
                # earlystop,
            ],
        )

        self.initialize_module()

        super(PUNPPCIClaimRegressor, self).fit(X, y)

        # Finished fitting!
        self.is_fitted_ = True

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """ Adds l1 regularization to the loss """
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        param_list = ["count_linear_0.data", "paid_linear_0.data"]
        loss += self.l1_l2_linear * sum(
            [
                w.abs().sum()
                for p, w in self.module_.named_parameters()
                if p in param_list
            ]
        )
        return loss

    def get_weights(self):
        """ Returns weights as a dictionary """
        return {p: v for p, v in self.module.named_parameters()}

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

    def predict(self, X, response_variable=None, w=None):
        X = (X - self.X_mean) / self.X_std

        if w is None:
            w = np.ones([X.shape[0], self.output_dimension * 2], np.float32)
        else:
            w = w.astype(np.float32) * np.ones(
                [X.shape[0], self.output_dimension * 2], np.float32
            )

        X = np.hstack([X, w]).astype(np.float32)

        pred = super(PUNPPCIClaimRegressor, self).predict(X)

        pred_df = pd.DataFrame(
            pred[:, 0 : self.output_dimension * 2],
            columns=self.claim_count_names + self.claim_paid_names,
        )

        # Return format
        if response_variable == "frequency":
            return pred_df[self.claim_count_names].sum(axis=0)
        elif response_variable == "size":
            return pred_df[self.claim_paid_names].sum(axis=0) / pred_df[
                self.claim_count_names
            ].sum(axis=0)
        else:
            return pred_df

    def _predict(self, X):
        return super(PUNPPCIClaimRegressor, self).predict(X)

    def score(self, X, y, w=None):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        X = (X - self.X_mean) / self.X_std

        if w is None:
            w = np.where(np.isnan(y), 0.0, 1.0)
        else:
            w = w * np.where(np.isnan(y), 0.0, 1.0)

        X = np.hstack([X, w]).astype(np.float32)

        pred = super(PUNPPCIClaimRegressor, self).predict(X)
        y_fill_na = np.where(np.isnan(y), 0.0, y)

        loss = (
            (pred[:, 0 : (self.output_dimension * 2)] - np.where(np.isnan(y), 0.0, y))
            ** 2
        ).mean()

        if np.isnan(loss):
            loss = np.finfo(np.float32).max

        print("Score: Loss = {}".format(loss))
        gc.collect()
        return -loss


class PUNPPCIClaimOptimizer(BayesSearchCV):
    def __init__(
        self,
        claim_count_names,
        claim_paid_names,
        feature_dimension=None,
        output_dimension=None,
        search_spaces={
            "l1_l2_linear": (1e-6, 1.0, "log-uniform"),
            "l2_residual": (1e-6, 1.0, "log-uniform"),
            "l2_bias": (1e-6, 1.0, "log-uniform"),
            "layer_size": [16, 64, 256, 1024],
        },
        weights=None,
        cv=5,
        n_iter=10,
        error_score=1e7,
    ):
        super(PUNPPCIClaimOptimizer, self).__init__(
            PUNPPCIClaimRegressor(
                claim_count_names=claim_count_names,
                claim_paid_names=claim_paid_names,
                feature_dimension=feature_dimension,
                output_dimension=output_dimension,
            ),
            search_spaces=search_spaces,
            cv=cv,
            n_iter=n_iter,
        )

    def fit(self, X, y, w=None):
        self.fit_params = {"w": w}
        super(PUNPPCIClaimOptimizer, self).fit(X, y, callback=self.on_step)

    def get_weights(self):
        """ Returns weights as a dictionary """
        return self.best_estimator_.get_weights()

    def linear_vs_residual(self):
        """ Compare linear effect vs residual effect """
        return self.best_estimator_.linear_vs_residual()

    def on_step(self, optim_result):
        """ Callback handler """
        print(
            """best score: {}, best params {}""".format(
                self.best_score_, self.best_params_
            )
        )
        # Clear Keras graph between each step
        # K.clear_session()

    def plot_graph(self):
        """ Plots the model graph to graphviz
        """
        return self.best_estimator_.plot_graph()

    def plot_pdp_frequency(self, dataset, feature):
        """ Explains frequency vs feature"""
        self.best_estimator_.preprocess = self.preprocess
        return self.best_estimator_.plot_pdp_frequency(dataset, feature)

    def plot_pdp_size(self, dataset, feature):
        """ Explains severity vs feature"""
        self.best_estimator_.preprocess = self.preprocess
        return self.best_estimator_.plot_pdp_size(dataset, feature)

    def predict(self, X, response_variable=None, *args, **kwargs):
        """ Predict the output or the expected cost, percentage developed

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
        return self.best_estimator_.predict(X, response_variable, *args, **kwargs)
