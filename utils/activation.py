import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pymc.math

def identity(x):
    return x

def relu(x):
    return pymc.math.maximum(0, x)
    #import theano.tensor as tt
    #return tt.nnet.relu(x)

def sigmoid(x):
    return pm.math.sigmoid(x)

def softmax(x):
    return pm.math.softmax(x)

def tanh(x):
    return pm.math.tanh(x)

def exp(x):
    return pm.math.exp(x)

def return_activation(activation: str):
    if activation in ('identity', None, 'linear'):
        return identity
    elif activation == 'relu':
        return relu
    elif activation == 'sigmoid':
        return sigmoid
    elif activation == 'softmax':
        return softmax
    elif activation == 'tanh':
        return tanh
    elif activation == 'exp':
        return exp
    else:
        raise NotImplementedError(f'Unknown activation {activation}')


if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)

    plt.ion()
    plt.plot(x, return_activation('identity')(x), label='identity')
    #plt.plot(x, return_activation('relu')(x), label='relu')
    plt.plot(x, return_activation('sigmoid')(x), label='sigmoid')
    plt.plot(x, return_activation('tanh')(x), label='tanh')
    plt.legend()

