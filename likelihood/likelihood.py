
import pymc as pm
from dataclasses import dataclass
from abc import abstractmethod
from typing import Dict

@dataclass
class Likelihood:

    @abstractmethod
    def build(self, params: Dict, observed):
        raise NotImplementedError('Missing build method')

@dataclass
class BernulliLikelihood(Likelihood):

    def build(self, params: Dict, observed):
        params = self._default_params(params)
        likelihood = pm.Bernoulli('likelihood', p=params['p'], shape=params['p'].shape, observed=observed)
        return likelihood

    def _default_params(self, params):
        if 'p' not in params:
            params['p'] = pm.Uniform('p', 0, 1)

@dataclass
class NormalLikelihood(Likelihood):


    def build(self, params: Dict, observed):
        params = self._default_params(params)
        likelihood = pm.Normal('likelihood', mu=params['mu'], sigma=params['sigma'], observed=observed, shape=params['mu'].shape)
        return likelihood

    def _default_params(self, params):
        if 'mu' not in params:
            params['mu'] = pm.Normal('mu', 0, 1)

        if 'sigma' not in params:
            params['sigma'] = pm.HalfNormal('sigma', 1)

        return params


