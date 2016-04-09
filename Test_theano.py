# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:54:40 2016

@author: shengx
"""

#%% Load Data

from Font import *
from utility import *
import numpy as np
import theano
from theano import tensor as T

with np.load('train_data.npz') as data:
    trainInput = data['trainInput']
    trainOutput = data['trainOutput']
    testInput = data['testInput']
    testOutput = data['testOutput']
    basis_size = int(data['basis_size'])
    testing_size = int(data['testing_size'])
    training_size = int(data['training_size'])


trainInput = trainInput.transpose()
trainOutput = trainOutput.transpose()
testInput = testInput.transpose()
testOutput = testOutput.transpose()    
batch_size = 50


def shared_dataset(data_x, data_y):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')

testInput, testOutput = shared_dataset(testInput, testOutput)
trainInput, trainOutput = shared_dataset(trainInput, trainOutput)     
#%% building neural networks

from NeuralNets import *

import numpy as np
import theano
from theano import tensor as T

rng = np.random.RandomState(1234)
nkerns = [5, 8]
learning_rate = 0.15


# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')
y = T.lvector('y')

print('...building the model')

layer0_input = x.reshape((batch_size, 1, basis_size, basis_size))

# first convolutional layer
# image original size 50X50, filter size 5X5, filter number nkerns[0]
# after filtering, image size reduced to (50 - 5 + 1) = 46
# after max pooling, image size reduced to 46 / 2 = 23
layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, basis_size, basis_size),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )
# second convolutional layer
# input image size 23X23, filter size 4X4, filter number nkerns[1]
# after filtering, image size (23 - 4 + 1) = 20
# after max pooling, image size reduced to 20 / 2 = 10    
layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
    
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=basis_size * basis_size,
        activation=T.nnet.sigmoid
    )
    
cost = ((layer2.output - y) ** 2).sum()

params = layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)

updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x: trainInput[index * batch_size: (index + 1) * batch_size],
            y: trainOutput[index * batch_size: (index + 1) * batch_size]
        }
    )    

test_model = theano.function(
        inputs = [index],
        outputs = cost,
        givens={
            x: testInput[index * batch_size: (index + 1) * batch_size],
            y: testOutput[index * batch_size: (index + 1) * batch_size]
        }
    )
#%%    

