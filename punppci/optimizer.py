from skopt import BayesSearchCV
from .model import PUNPPCILossEstimator
from keras import backend as K


class PUNPPCILossOptimizer(BayesSearchCV):
    def __init__(
        self,
        dataset,
        search_spaces={
            "epochs": (1, 10),
            "l1_l2_lin_pricing": (1e-6, 1e6, "log-uniform"),
            "l1_l2_res_pricing": (1e-2, 1e6, "log-uniform"),
            "dense_layers_pricing": (1, 5),
            "dense_size_pricing": [16, 64, 256],
            "l1_l2_lin_development": (1e-6, 1e6, "log-uniform"),
            "l1_l2_res_development": (1e-2, 1e6, "log-uniform"),
            "dense_layers_development": (1, 5),
            "dense_size_development": [16, 64, 128],
            "clipnorm": (1, 10, "log-uniform"),
        },
        weights=None,
        cv=4,
        n_iter=10,
    ):
        super(PUNPPCILossOptimizer, self).__init__(
            PUNPPCILossEstimator(dataset), search_spaces
        )

    def fit(self, X, y, w=None):
        self.fit_params = {"w": w, "verbose": 0}
        super(PUNPPCILossOptimizer, self).fit(X, y, callback=self.on_step)

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
        return self.best_estimator_.plot_pdp_frequency(dataset, feature)

    def plot_pdp_size(self, dataset, feature):
        """ Explains severity vs feature"""
        return self.best_estimator_.plot_pdp_size(dataset, feature)

    def ppci(self, dataset):
        """ Tabulates a triangle of PPCI and the selections
        """
        return self.best_estimator_.ppci(dataset)

    def print_layers(self):
        """ Print layers and shapes """
        return self.best_estimator_.print_layers()

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
