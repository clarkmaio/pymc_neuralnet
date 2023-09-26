# Pymc Neuralnet
Simple package to build Bayesian neuralnets with Keras-like syntax.

Use layers and `add` method to build a neural to model Likelihood parameters.
When adding a layer the first parameter of the `.add` method of the `Sequential` model is the name of the parameter of the Likelihood that will be modeled by the nerual net.

<br>
You can assign a likelihood function using the `.add_likelihood` and `Likelihood` class or use Sequential model with pre-assigned likelihood:

| Sequential model | Parameters list |
|----------|---------|
|`NormalSequential`|`'mu'`, `'sigma'`|
|`BernulliSequential`|`'p'`|


```
# Example: NormalSequential has normal Likelihood.
# You can model 'mu', 'sigma' parameters.
# If not specified a defualt prior distribution will be used

model = NormalSequential()
model.add('mu', Input(input_size=X.shape[1]))
mdl.add('mu', Input(input_size=X.shape[1]))
mdl.add('mu', Dense(units=10, use_bias=True, activation='tanh'))
mdl.add('mu', Dense(units=1, use_bias=True, activation='linear'))
```

See `doc/example.ipynb` for an example.

![spline](https://github.com/clarkmaio/pymc_neuralnet/blob/master/doc/img/spline.png)

![trace](https://github.com/clarkmaio/pymc_neuralnet/blob/master/doc/img/trace.png)


<br><br>

:warning: **Todo:**
* :x: Better documentation and more examples!!!!!
* :white_check_mark: Sequential model
* :white_check_mark: NormalSequential model
* :x: BernulliSequential model
* :x: Non Sequential model
* :x: Integrate optuna to perform gridsearch

