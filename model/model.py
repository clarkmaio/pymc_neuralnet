
from dataclasses import dataclass
import pymc as pm
from typing import Iterable, Dict, List
import pandas as pd
from pymc_neuralnet.layers.layer import Layer
from pymc_neuralnet.layers.layer import Input, Dense
from pymc_neuralnet.likelihood.likelihood import Likelihood, NormalLikelihood
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

@dataclass
class Model:
    layers: List = None

    def __post_init__(self):
        self._model = pm.Model()
        self._trace = None
        self._approx = None


        self._X_data_train = None
        self._y_data_train = None

        if self.layers is None:
            self.layers = {}

        self._likelihood_params = {}

    @property
    def trace(self):
        return self._trace

    @property
    def approx(self):
        return self._approx

    @property
    def likelihood_params(self):
        return self._likelihood_params

    def set_likelihood_param(self, param, distribution: pm.Distribution):
        self._likelihood_params[param] = distribution

    def _layer_check(self, layers: Iterable):
        assert isinstance(layers[0], Input), 'First layer must be Input layer'

    def forward(self, layers: Iterable):
        '''Concatenate layers'''

        self._layer_check(layers=layers)

        output = layers[0](self._X_data_train)
        with self._model:
            for l in layers[1:]:
                output = l(output)
        return output

    def _build_likelihood(self, likelihood: Likelihood, params, observed):
        return likelihood.build(params=params, observed=observed)

    @staticmethod
    def fit(self, X, y, use_advi: bool = False, sample_kws: Dict = {}, advi_kws: Dict = {}):
        raise NotImplementedError('fit method')

    def fit_sample(self, use_advi:bool = False, sample_kws = {}, advi_kws = {}):
        if use_advi:
            with self._model:
                self._approx = pm.fit(**advi_kws)
                self._trace = self._approx.sample(**sample_kws)
        else:
            with self._model:
                self._trace = pm.sample(**sample_kws)

    def predict(self, X, quantile: Iterable = None, return_sample: bool = False):
        '''

        :param X:
        :param return_mean:
        :param quantile:
        :return:
        '''
        with self._model:
            pm.set_data({'X': X})
            ppc = pm.sample_posterior_predictive(trace=self._trace)

            if return_sample:
                return ppc.posterior_predictive.likelihood
            elif quantile is None:
                return ppc.posterior_predictive.likelihood.mean(dim=['chain', 'draw']).values
            else:
                return ppc.posterior_predictive.likelihood.quantile(q=quantile, dim=['chain', 'draw']).values.T




    def plot_trace(self, **kwargs):
        plt.ion()
        az.plot_trace(data=self._trace, **kwargs)
        plt.tight_layout()

    def plot_ELBO(self):
        plt.ion()
        plt.figure()
        plt.plot(self._approx.hist, color='blue', alpha=1)
        plt.grid(linestyle=':')
        plt.title('ELBO', fontweight='bold')
        plt.xlabel('Iterations')
        plt.ylabel('ELBO')
        plt.tight_layout()

    def plot_forest(self, **kwargs):
        plt.ion()
        az.plot_forest(self._trace, **kwargs)
        plt.tight_layout()
    
    def plot_graph(self):
        plt.ion()
        pm.model_to_graphviz(model=self._model)
    
    def summary(self):
        return az.summary(data=self._trace)



@dataclass
class Sequential(Model):

    def __post_init__(self):
        super().__post_init__()

    def add(self, label: str, layer: Layer):
        '''
        If it is the first time you add a Layer referring to a label initialize the list
        Otherwise just append layer to the list of layer referring to label variable
        '''

        layer.set_label(label=label)

        if label in self.layers.keys():
            self.layers[label].append(layer)
        else:
            self.layers[label] = [layer]

    def add_likelihood(self, likelihood: Likelihood):
        self._likelihood_fun = likelihood

    def fit(self, X, y, use_advi: bool = False, sample_kws: Dict = {}, advi_kws: Dict = {}):

        with self._model:
            self._X_data_train = pm.MutableData('X', value=X)
            self._y_data_train = pm.MutableData('y', value=y)

        for k in self.layers:
            output = self.forward(layers = self.layers[k])

        with self._model:
            self._likelihood_params[k] = output._output

        with self._model:
            self._likelihood = self._build_likelihood(likelihood=self._likelihood_fun, params=self._likelihood_params, observed=self._y_data_train)
        self.fit_sample(use_advi=use_advi, sample_kws=sample_kws, advi_kws=advi_kws)


@dataclass
class NormalSequential(Sequential):


    def __post_init__(self):
        super().__post_init__()
        self.add_likelihood(likelihood=NormalLikelihood())

    def fit(self, X, y, use_advi: bool = False, sample_kws: Dict = {}, advi_kws: Dict = {}):

        # Define params in case they are missing
        if ('sigma' not in self.layers) and ('sigma' not in self._likelihood_params):
            with self._model:
                self._likelihood_params['sigma'] = pm.HalfNormal('sigma', 1)

        if ('mu' not in self.layers) and ('mu' not in self._likelihood_params):
            with self._model:
                self._likelihood_params['mu'] = pm.Normal('mu', 0, 1)

        super().fit(X=X, y=y, use_advi=use_advi, sample_kws=sample_kws, advi_kws=advi_kws)


if __name__ == '__main__':

    N = 1000
    X = np.linspace(0, 5, N)
    y = X**2 + 100*np.sin(X) + np.random.randn(N) * 30.-100

    # Minmax scaling y
    y = (y - y.min()) / (y.max() - y.min())

    # Traib test split
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=.8)

    # Build Neural net
    mdl = NormalSequential()
    mdl.add('mu', Input(input_size=X.shape[1]))
    mdl.add('mu', Dense(units=10, use_bias=True, activation='tanh'))
    mdl.add('mu', Dense(units=1, use_bias=True, activation='linear'))
    mdl.fit(X_train, y_train, use_advi=True, advi_kws={'n': 100000}, sample_kws={'draws': 5000})

    # Prediction
    y_pred = mdl.predict(X=X_test)
    y_pred_quantile = mdl.predict(X=X_test, quantile=[.025, .975])


    # Plot result
    plt.ion()
    pred_df = pd.DataFrame({'x': X_test[:, 0], 'y1': y_pred_quantile[:, 0], 'y2': y_pred_quantile[:, 1], 'y': y_pred}).sort_values(by='x')
    plt.scatter(X_test[:, 0], y_test, color='black', alpha=.3, label='Test')
    plt.plot(pred_df['x'], pred_df['y'], color='blue', alpha=1, label='Test prediction')
    plt.fill_between(pred_df['x'], pred_df['y1'], pred_df['y2'], color='blue', alpha=.1, label='Test prediction 95% CI')
    plt.legend()
    plt.grid(linestyle=':')
    plt.show(block=True)

    plt.savefig('/home/clarkmaio/Scrivania/plot.png')
    plt.close()

