import re
from keras.layers import (
    Input,
    Embedding,
    Lambda,
    Dense,
    Reshape,
    Layer,
    concatenate,
    add,
)
from keras import regularizers
from keras.constraints import Constraint
from keras.initializers import Zeros
from keras import backend as K


class NonNegativeSumOne(Constraint):
    """ Constrains the weights incident to each hidden unit to be
        non-negative and sum to one, and keep to a single value.
    # Arguments
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        # Non Negative
        w *= K.cast(K.greater_equal(w, 0.0), K.floatx())

        # Scale to sum to one
        sc = w / (K.epsilon() + K.sum(w, axis=self.axis, keepdims=True))
        return sc

    def get_config(self):
        return {"axis": self.axis}


class NonNegativeWeightedAverage(Layer):
    """ Layer that returns a non negative weighted average, with the same
    weights for the whole vector"""

    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(NonNegativeWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(2,),
            initializer="uniform",
            constraint=NonNegativeSumOne(),
            trainable=True,
        )
        super(NonNegativeWeightedAverage, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return a * self.kernel[0] + b * self.kernel[1]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], self.output_dim)


"""
PenalisedUnexplainabilityNetwork
 - Linear Factors
 - "Unexplained" Residual Neural Network
 - l1 and l2 penalties for neural network complexity

Outputs
 - Ultimate claim count: Poisson
 - PPCI at Time Delay: (multiply by counts): MSE
 - Ult cost, mean and variance of delay: MSE, Negative Log Likelihood
    [the latter would require different data structure?]
"""


def network_inputs(variate_features, categorical_features):
    """ The required variate and categorical inputs and relevant responses

    Parameters:
    -----------
    variate_features: List of numerical input feature names
    categorical_features: List of categorical input feature names

    """

    variate_inputs = [Input(shape=(1,), name=x) for x in variate_features]
    categorical_inputs = [Input(shape=(1,), name=x) for x in categorical_features]

    return variate_inputs, categorical_inputs


def network_embeddings(
    variate_inputs,
    categorical_inputs,
    categorical_dimensions,
    embedding_dim=1,
    name_prefix="input_embed_",
):
    """ Add embeddings

    Parameters:
    -----------
        variate_inputs: List of numerical inputs
        categorical_inputs: List of categorical inputs
        categorical_dimensions: List of corresponding dimensions of the categorical
        embedding_dim: Size of categorical embeddings

    Returns: inputs_with_embeddings
    """
    categorical_embeddings = [
        Reshape(target_shape=(embedding_dim,))(
            Embedding(
                output_dim=embedding_dim,
                input_dim=n,
                embeddings_initializer=Zeros(),
                # embeddings_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                name=re.sub(r"\W+", "", name_prefix + x.name),
            )(x)
        )
        for x, n in zip(categorical_inputs, categorical_dimensions)
    ]

    inputs_with_embeddings = concatenate(categorical_embeddings + variate_inputs)

    return inputs_with_embeddings


def penalised_unexplainability_network(
    inputs,
    response,
    name_prefix="",
    l1_lin=0.01,
    l2_lin=0.01,
    l1_res=0.01,
    l2_res=0.01,
    dense_layers=3,
    dense_size=64,
    n_dim=1,
    bias_initializer="he_uniform",
    activation="linear",
):

    """ Returns a keras network of the explainable + linear model

    Parameters
    ----------

    response: List of responses
    l1: l1 regularizer
    l2: l2 regularizer
    name_prefix: prefix to add to the network layers

    dense_layers: Number of unexplainable AI features
    dense_size: Size of unexplainable AI layers

    Returns
    -------
    keras output layers of the network

    """

    # Linear outputs
    linear_outputs = [
        Dense(
            n_dim,
            activation=activation,
            kernel_initializer=Zeros(),
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l1_l2(l1=l1_lin, l2=l2_lin),
            name="{}linear_output_{}".format(name_prefix, response_name),
        )(inputs)
        for response_name in response
    ]

    # Residual Network (concat version)
    resnet = inputs

    for i in range(0, dense_layers):
        dense = Dense(
            dense_size,
            activation="selu",
            kernel_regularizer=regularizers.l1_l2(l1=l1_res, l2=l2_res),
            name="{}residual_layer_{}".format(name_prefix, i),
        )(inputs)

        resnet = concatenate([dense, resnet])

    residual_outputs = [
        Dense(
            n_dim,
            activation=activation,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l1_l2(l1=l1_res, l2=l2_res),
            name="{}residual_output_{}".format(name_prefix, response_name),
        )(resnet)
        for response_name in response
    ]

    # Output
    outputs = [
        NonNegativeWeightedAverage(
            output_dim=n_dim,
            name="{}blend_weight_{}".format(name_prefix, response_name),
        )([lin, res])
        for lin, res, response_name in zip(linear_outputs, residual_outputs, response)
    ]

    return outputs
