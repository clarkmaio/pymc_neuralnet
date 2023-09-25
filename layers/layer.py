from dataclasses import dataclass
import pymc as pm
import numpy as np
from ..utils.activation import return_activation

@dataclass
class Layer:
    units: int
    label = None

    def __post_init__(self):
        self._level = None
        self._output = None
        return

    def set_label(self, label: str):
        self.label = label


class Input(Layer):

    def __init__(self, input_size: int):
        self.input_size = input_size
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        self.units = self.input_size
        self._level = 0

    def __call__(self, X):
        self._output = X
        return self


class Dense(Layer):

    def __init__(self, units,
                 label = None,
                 activation = None,
                 use_bias: bool = True):

        self.units = units
        self.label = label
        self.activation = activation
        self.use_bias = use_bias
        self.__post_init__()


    def __post_init__(self):
        super().__post_init__()
        self._W = None
        self._b = 0
        self._activation_fun = None


    def __call__(self, layer: Layer):

        # Define layer id
        self._level = layer._level + 1

        # Initialize weights
        if self.units == 1:
            shape = layer.units
            initval = np.random.randn(shape)
            self._W = pm.Normal(f'{self.label}_W_{self._level}', mu=0, sigma=1, shape = shape, initval=initval)
        else:
            shape = (layer.units, self.units)
            initval = np.random.randn(*shape)
            self._W = pm.Normal(f'{self.label}_W_{self._level}', mu=0, sigma=1, shape = shape, initval=initval)


        # Initialize Bias
        if self.use_bias:
            self._b = pm.Normal(f'{self.label}_b_{self._level}', mu=0, sigma=1, initval=np.random.randn())
        else:
            self._b = 0

        # Activation function
        self._activation_fun = return_activation(activation = self.activation)

        # Output
        self._output = self._activation_fun(
            pm.math.dot(layer._output, self._W) + self._b
        )

        return self

