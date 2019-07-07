import torch
from torch import nn
import torch.nn.functional as F
from warnings import warn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hiddenlayer as hl
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, LRScheduler, EarlyStopping, GradientNormClipping
from skorch.callbacks.lr_scheduler import CyclicLR
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn import preprocessing
from skopt import BayesSearchCV
from pdpbox import pdp
import gc

try:
    import databricks.koalas as kl
except ImportError:
    import pandas as kl

    pass

EPSILON = 0.00001


default_device = "cuda" if torch.cuda.is_available() else "cpu"


class PUNPPCIClaimModule(nn.Module):
    def __init__(
        self,
        feature_dim=None,
        output_dim=None,
        cat_dim=None,
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

        # Categorical dimensions (column loc, number of levels)
        self.cat_dim = cat_dim

        if self.cat_dim is not None:
            # 1d Embeddings Linear
            self.embeddings_linear_count = nn.ModuleList(
                [nn.Embedding(lvls + 1, 1) for loc, lvls in cat_dim]  # lvls
            )

            self.embeddings_linear_paid = nn.ModuleList(
                [nn.Embedding(lvls + 1, 1) for loc, lvls in cat_dim]  # lvls
            )

            # 2d Embeddings for Neural Network
            # with 4th root rule https://www.tensorflow.org/guide/feature_columns
            self.embeddings_residual = nn.ModuleList(
                [
                    nn.Embedding(lvls + 1, int(np.ceil(lvls ** 0.25)))
                    for loc, lvls in cat_dim
                ]  # lvls
            )

            self.n_embed = max([loc for loc, lvls in cat_dim])

            extra_embed_dim = int(
                sum([(np.ceil(lvls ** 0.25) - 1) for loc, lvls in cat_dim])
            )

            for l in self.embeddings_linear_count:
                nn.init.zeros_(l.weight.data)

            for l in self.embeddings_linear_paid:
                nn.init.zeros_(l.weight.data)

            for l in self.embeddings_residual:
                nn.init.kaiming_normal_(l.weight.data)

        else:
            self.n_embed = 0

        # self.l1_l2_lin_pricing = l1_l2_lin_pricing
        # self.l1_l2_res_pricing = l1_l2_res_pricing

        # Pricing - Neural Network
        self.non_lin_0 = nonlin
        self.non_lin_1 = nonlin
        self.non_lin_2 = nonlin

        self.dense_pricing_0 = nn.Linear(feature_dim + extra_embed_dim, layer_size)
        self.dense_pricing_0_bn = nn.BatchNorm1d(layer_size)
        self.dense_pricing_1 = nn.Linear(layer_size, layer_size)
        self.dense_pricing_1_bn = nn.BatchNorm1d(layer_size)
        self.dense_pricing_2 = nn.Linear(layer_size, layer_size)
        self.dense_pricing_2_bn = nn.BatchNorm1d(layer_size)

        # Pricing - Neural Network Output
        self.dropout = nn.Dropout(0.5)
        self.count_residual_0 = nn.Linear(layer_size, 1)
        self.paid_residual_0 = nn.Linear(layer_size, 1)

        # Pricing - Linear
        self.count_linear_0 = nn.Linear(feature_dim - self.n_embed, 1)
        self.paid_linear_0 = nn.Linear(feature_dim - self.n_embed, 1)

        # Pricing - Blend
        self.count_blend_w = nn.Parameter(torch.Tensor(1))
        self.paid_blend_w = nn.Parameter(torch.Tensor(1))

        # Reserving - Spread
        self.count_spread = nn.Parameter(torch.Tensor(output_dim))
        self.paid_spread = nn.Parameter(torch.Tensor(output_dim))

        self.count_residual_spread = nn.Linear(
            layer_size + feature_dim + extra_embed_dim,
            output_dim,  # layer_size + feature_dim + extra_embed_dim
        )
        self.paid_residual_spread = nn.Linear(
            layer_size + feature_dim + extra_embed_dim,
            output_dim,  # layer_size + feature_dim + extra_embed_dim
        )

        # Initialise GLM components weights at zero
        nn.init.zeros_(self.count_linear_0.weight)
        nn.init.zeros_(self.paid_linear_0.weight)

        # Initialise GLM components biases with priors
        if y_mean is None:
            nn.init.normal(self.count_spread.data, mean=0, std=0.1)
            nn.init.normal(self.paid_spread.data, mean=0, std=0.1)
        else:
            # print(y_mean)

            freq_mean = y_mean[0:output_dim] + EPSILON
            paid_mean = y_mean[output_dim : 2 * output_dim] + EPSILON

            # print("Initial frequency: {}, paid: {}".format(total_freq, total_paid))

            self.count_spread.data = torch.Tensor(np.log(freq_mean))
            self.count_linear_0.bias.data = torch.Tensor(
                np.log(freq_mean.sum(keepdims=True))
            )
            self.paid_spread.data = torch.Tensor(np.log(paid_mean) - np.log(freq_mean))
            self.paid_linear_0.bias.data = torch.Tensor(
                np.log(paid_mean.sum(keepdims=True))
                - np.log(freq_mean.sum(keepdims=True))
            )

        # Initialise Neural Network Output weights
        nn.init.kaiming_normal_(self.dense_pricing_0.weight)
        nn.init.kaiming_normal_(self.dense_pricing_1.weight)
        nn.init.kaiming_normal_(self.dense_pricing_2.weight)

        nn.init.kaiming_normal_(self.count_residual_spread.weight)
        nn.init.kaiming_normal_(self.paid_residual_spread.weight)

        nn.init.zeros_(self.dense_pricing_0.bias)
        nn.init.zeros_(self.dense_pricing_1.bias)
        nn.init.zeros_(self.dense_pricing_2.bias)

        nn.init.zeros_(self.count_residual_0.bias)
        nn.init.zeros_(self.paid_residual_0.bias)

        nn.init.zeros_(self.count_residual_spread.bias)
        nn.init.zeros_(self.paid_residual_spread.bias)

        # Initialise weights at 50%
        nn.init.zeros_(self.count_blend_w.data)
        nn.init.zeros_(self.paid_blend_w.data)

    def forward(self, X, **kwargs):

        # Split
        X, wc, wp = list(
            torch.split(X, [self.feature_dim, self.output_dim, self.output_dim], dim=1)
        )

        # Split more if categories to apply
        if self.cat_dim is not None:

            Xx = torch.split(
                X,
                [1] * (self.n_embed + 1) + [self.feature_dim - (self.n_embed + 1)],
                dim=1,
            )

            # Linear embedding weights

            embeds_lin_count = [
                emb(Xxx.type(torch.LongTensor))
                for Xxx, emb in zip(
                    Xx[1 : (self.n_embed + 1)], self.embeddings_linear_count
                )
            ]

            Xelc = torch.squeeze(torch.cat(embeds_lin_count, 2), dim=1)

            embeds_lin_paid = [
                emb(Xxx.type(torch.LongTensor))
                for Xxx, emb in zip(
                    Xx[1 : (self.n_embed + 1)], self.embeddings_linear_paid
                )
            ]

            Xelp = torch.squeeze(torch.cat(embeds_lin_paid, 2), dim=1)

            embeds_res = [
                emb(Xxx.type(torch.LongTensor))
                for Xxx, emb in zip(
                    Xx[1 : (self.n_embed + 1)], self.embeddings_residual
                )
            ]

            # Deep embedding weights
            Xer = torch.squeeze(torch.cat(embeds_res, 2), dim=1)

            Xl = torch.cat((Xx[0], Xx[-1]), 1)

            Xr = torch.cat((Xx[0], Xer, Xx[-1]), 1)

            # Linear
            count_linear = self.count_linear_0(Xl) + Xelc.sum(1).unsqueeze(1)
            paid_linear = (
                count_linear + self.paid_linear_0(Xl) + Xelp.sum(1).unsqueeze(1)
            )

        else:
            # Linear Frequency and Risk Premium
            count_linear = self.count_linear_0(Xl)
            paid_linear = count_linear + self.paid_linear_0(Xl)

        # Resnet
        Xr0 = self.dropout(Xr)
        Xr1 = self.non_lin_0(self.dense_pricing_0_bn(self.dense_pricing_0(Xr0)))
        Xr2 = self.non_lin_1(self.dense_pricing_1_bn(self.dense_pricing_1(Xr1)))
        Xr3 = self.non_lin_2(self.dense_pricing_2_bn(self.dense_pricing_2(Xr2)))

        # Resnet Frequency and Risk Premium
        count_residual = self.count_residual_0(Xr3)
        paid_residual = count_residual + self.paid_residual_0(Xr3)

        # Spread - Constant
        count_s = self.count_spread
        paid_s = count_s + self.paid_spread

        count_dev_linear = wc * torch.exp(count_linear + F.log_softmax(count_s))
        paid_dev_linear = wp * torch.exp(paid_linear + F.log_softmax(paid_s))

        # Spread - Residual
        count_sr = self.count_residual_spread(torch.cat((Xr3, Xr0), 1))
        paid_sr = count_sr + self.paid_residual_spread(torch.cat((Xr3, Xr0), 1))

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
                print("No gradient for:")
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

    def on_epoch_begin(self, net, **kwargs):
        # Test every 10 epochs
        if net.history[-1]["epoch"] % self.modulo == 0:
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
        claim_count_names=None,
        claim_paid_names=None,
        feature_dimension=None,
        output_dimension=None,
        categorical_dimensions=None,
        # Calibrate these parameters:
        layer_size=100,
        l1_l2_linear=0.001,
        l2_weights_residual=0.01,
        l2_bias_residual=0.01,
        device=default_device,
        # Leave these parameters alone generally:
        batch_size=5000,
        optimizer=torch.optim.Adam,
        max_epochs=100,
        lr_range=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
        patience=2,
        clipnorm=0.1,

    ):
        self.device = device

        self.optimizer = optimizer

        self.feature_dimension = feature_dimension
        self.output_dimension = output_dimension
        self.categorical_dimensions = categorical_dimensions
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        self.layer_size = layer_size

        self.lr_range = lr_range
        self.l1_l2_linear = l1_l2_linear
        self.l2_weights_residual = l2_weights_residual
        self.l2_bias_residual = l2_bias_residual
        self.patience = patience
        self.clipnorm = clipnorm

        self.claim_count_names = claim_count_names
        self.claim_paid_names = claim_paid_names

        # For the BayesSearchCV
        self.initialized_ = True

    def fit(self, X, y, *args, w=None, **kwargs):

        # Determine optional parameters
        if self.claim_count_names is None:
            self.claim_count_names = [
                "claim_count_{}".format(x) for x in range(0, int(y.shape[1] / 2))
            ]

        if self.claim_paid_names is None:
            self.claim_paid_names = [
                "claim_paid_{}".format(x) for x in range(0, int(y.shape[1] / 2))
            ]

        if self.feature_dimension is None:
            self.feature_dimension = X.shape[1]

        if self.output_dimension is None:
            self.output_dimension = len(self.claim_paid_names)

        if self.categorical_dimensions is None:
            self.categorical_dimensions = []
            # TODO: This is a bit slow and unstable, is there a better way?
            for i in range(X.shape[1]):
                X_int = X[:, i].astype(int)
                if np.all((X_int - X[:, i]) == 0):
                    self.categorical_dimensions += [(i, np.max(X_int))]
            print(
                "Auto detect categorical dimensions to be: {}".format(
                    self.categorical_dimensions
                )
            )
        # Standardize outputs
        # self.X_mean = np.mean(X, axis=0)
        # self.X_std = np.std(X, axis=0)

        # Except categoricals
        # for i, j in self.categorical_dimensions:
        #     self.X_mean[i] = 0
        #     self.X_std[i] = 1

        # X = (X - self.X_mean) / self.X_std

        # Shuffle X, y
        X, y = shuffle(X, y, random_state=0)

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
            print("NOTE: Data size is small, outcomes may be odd.")
            batch_size = X.shape[0]
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
                    cat_dim=self.categorical_dimensions,
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
                    ("embeddings_linear*", {"weight_decay": self.l1_l2_linear}),
                    (
                        "embeddings_residual*",
                        {"weight_decay": self.l2_weights_residual},
                    ),
                    ("dense_pricing*", {"weight_decay": self.l2_weights_residual}),
                    ("count_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                    ("paid_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                    (
                        "count_residual_spread.bias",
                        {"weight_decay": self.l2_bias_residual},
                    ),
                    (
                        "paid_residual_spread.bias",
                        {"weight_decay": self.l2_bias_residual},
                    ),
                    ("count_residual_0.bias", {"weight_decay": self.l2_bias_residual}),
                    ("paid_residual_0.bias", {"weight_decay": self.l2_bias_residual}),
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
        expected_epoch_count = 4 if expected_epoch_count < 4 else expected_epoch_count

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
                cat_dim=self.categorical_dimensions,
                y_mean=y_mean,
                layer_size=self.layer_size,
            ),
            max_epochs=expected_epoch_count,
            lr=self.lr_min,
            optimizer=self.optimizer,
            # optimizer__momentum=self.momentum,
            optimizer__param_groups=[
                ("embeddings_linear*", {"weight_decay": self.l1_l2_linear}),
                ("embeddings_residual*", {"weight_decay": self.l2_weights_residual}),
                ("dense_pricing*", {"weight_decay": self.l2_weights_residual}),
                ("count_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                ("paid_linear_0.weight", {"weight_decay": self.l1_l2_linear}),
                ("count_residual_spread.bias", {"weight_decay": self.l2_bias_residual}),
                ("paid_residual_spread.bias", {"weight_decay": self.l2_bias_residual}),
                ("count_residual_0.bias", {"weight_decay": self.l2_bias_residual}),
                ("paid_residual_0.bias", {"weight_decay": self.l2_bias_residual}),
            ],
            batch_size=batch_size,
            criterion=nn.MSELoss,
            callbacks=[
                CheckNaN(),
                # CheckMean(X, self.output_dimension, 1),  # expected_epoch_count
                cyclic_lr,
                gradclip,
                # earlystop,
            ],
        )

        self.initialize_module()

        super(PUNPPCIClaimRegressor, self).fit(X, y)

        # Finished fitting!
        self.is_fitted_ = True

    def get_datasets_format(self):
        return (
            ["origin", "features"],
            ["claim_count", "claim_paid"],
            self.preprocess_X,
            None,
        )

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

    def linear_vs_residual(self):
        """ Compare linear effect vs residual effect """
        weights = self.get_weights()
        return weights["count_blend_w"], weights["paid_blend_w"]

    def plot_pdp_1d(self, dataset, feature, response):
        """ Explains response vs feature """
        if isinstance(dataset.features, pd.DataFrame):
            pd0 = pd
        elif isinstance(dataset.features, kl.DataFrame):
            pd0 = kl

        features = pd0.concat([dataset.origin, dataset.features], axis=1)

        features.iloc[:, 0] = features.iloc[:, 0].values.astype(float).astype("float32")
        # Use transformed df if categorical, otherwise
        # use the untransformed dataset with a transformer
        if hasattr(features[feature], "cat"):
            # Transformed values
            df = pd.DataFrame(
                self.preprocess.transform(features.to_numpy()), columns=features.columns
            )

            def data_transformer(df):
                return df

        else:
            df = features

            def data_transformer(df):
                return pd0.DataFrame(
                    self.preprocess.transform(df.to_numpy()), columns=features.columns
                )

        pdp_iso = pdp.pdp_isolate(
            model=self,
            dataset=df,
            predict_kwds={"response_variable": response},
            model_features=features.columns,
            data_transformer=data_transformer,
            feature=feature,
        )

        fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_iso, feature_name=feature)

        if hasattr(features[feature], "cat"):
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

    def plot_graph(self):
        """ Plots network graph.

        If graph is too small, consider:

        from IPython.display import HTML
        style = "<style>svg{width:50% !important;height:50% !important;}</style>"
        HTML(style)
        """
        return hl.build_graph(
            self.module,
            torch.ones([1, self.feature_dimension + self.output_dimension * 2]),
        )

    def preprocess_X(self, features):
        def converter(x):
            if x.dtype == "object":
                return x.astype("category")
            elif x.dtype == "int64":
                return x.astype("float32")
            elif x.dtype == "float64":
                return x.astype("float32")
            else:
                return x

        features_convert = features.apply(converter)

        # Transformer - Features
        self.origin_features = features_convert.columns == features_convert.columns[0]
        self.categorical_features = features_convert.dtypes == "category"
        self.numerical_features = ~(self.categorical_features | self.origin_features)

        features_convert.iloc[:, 0] = (
            features_convert.iloc[:, 0].values.astype(float).astype("float32")
        )

        if hasattr(self, "preprocess"):
            results = self.preprocess.transform(features)
        else:  # Alternative was OneHotEncoder(sparse=False) for categorical_features
            self.preprocess = make_column_transformer(
                (preprocessing.MinMaxScaler(), self.origin_features),
                (preprocessing.OrdinalEncoder(), self.categorical_features),
                (preprocessing.StandardScaler(), self.numerical_features),
            )

            results = self.preprocess.fit_transform(features)

        return results

    def predict(self, X, response_variable=None, w=None):
        # X = (X - self.X_mean) / self.X_std

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
            return pred_df[self.claim_count_names].sum(axis=1)
        elif response_variable == "size":
            return pred_df[self.claim_paid_names].sum(axis=1) / pred_df[
                self.claim_count_names
            ].sum(axis=1)
        elif response_variable == "dataframe":
            return pred_df
        else:
            return pred_df.to_numpy()

    def _predict(self, X):
        return super(PUNPPCIClaimRegressor, self).predict(X)

    def score(self, X, y, w=None):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        # X = (X - self.X_mean) / self.X_std

        if w is None:
            w = np.where(np.isnan(y), 0.0, 1.0)
        else:
            w = w * np.where(np.isnan(y), 0.0, 1.0)

        X = np.hstack([X, w]).astype(np.float32)

        pred = super(PUNPPCIClaimRegressor, self).predict(X)

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
        claim_count_names=None,
        claim_paid_names=None,
        feature_dimension=None,
        output_dimension=None,
        categorical_dimensions=None,
        search_spaces={
            "l1_l2_linear": (1e-6, 0.1, "log-uniform"),
            "l2_weights_residual": (1e-4, 0.1, "log-uniform"),
            "l2_bias_residual": (1e-3, 0.1, "log-uniform"),
            "layer_size": [64, 256, 1024],
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
                categorical_dimensions=categorical_dimensions,
            ),
            search_spaces=search_spaces,
            cv=cv,
            n_iter=n_iter,
        )

    def fit(self, X, y, w=None):
        self.fit_params = {"w": w}
        super(PUNPPCIClaimOptimizer, self).fit(X, y, callback=self.on_step)

    def get_datasets_format(self):
        return (
            ["origin", "features"],
            ["claim_count", "claim_paid"],
            self.preprocess_X,
            None,
        )

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

    def preprocess_X(self, features):
        def converter(x):
            if x.dtype == "object":
                return x.astype("category")
            elif x.dtype == "int64":
                return x.astype("float32")
            elif x.dtype == "float64":
                return x.astype("float32")
            else:
                return x

        features_convert = features.apply(converter)

        # Transformer - Features
        origin_features = features_convert.columns == features_convert.columns[0]
        categorical_features = features_convert.dtypes == "category"
        numerical_features = ~(categorical_features | origin_features)

        features_convert.iloc[:, 0] = (
            features_convert.iloc[:, 0].values.astype(float).astype("float32")
        )

        if hasattr(self, "preprocess"):
            results = self.preprocess.transform(features)
        else:
            self.preprocess = make_column_transformer(
                (preprocessing.MinMaxScaler(), origin_features),
                (preprocessing.OrdinalEncoder(), categorical_features),
                (preprocessing.StandardScaler(), numerical_features),
            )

            results = self.preprocess.fit_transform(features)

        return results

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
