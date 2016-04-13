# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:54:40 2016

@author: shengx
"""

#%% Load Data

import numpy as npFont
import theano
from theano import tensor as T


from Fontv2 import *
from utility import *
from NeuralNets import *
basis_size = 36
font_dir = '/home/shengx/Documents/Font'
input_letter = ['B','A','S','Q']
output_letter = ['R']

Fonts = Font(basis_size, font_dir, input_letter, output_letter )
trainInput, trainOutput, testInput, testOutput = Fonts.getLetterSets(1000,200)
trainInput = 1 - trainInput
trainOutput = 1 - trainOutput
testInput = 1 - testInput
testOutput = 1 - testOutput

trainInput, trainOutput = shared_dataset(trainInput, trainOutput) 


batch_size = 1

#%% building neural networks



rng1 = np.random.RandomState(1234)
rng2 = np.random.RandomState(2345)
rng3 = np.random.RandomState(1567)
rng4 = np.random.RandomState(1124)
nkerns = [8, 10]
learning_rate = 1


# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')
y = T.imatrix('y')

print('...building the model')

layer0_input = x.reshape((batch_size, 1, basis_size, basis_size))

# first convolutional layer
# image original size 50X50, filter size 5X5, filter number nkerns[0]
# after filtering, image size reduced to (50 - 5 + 1) = 46
# after max pooling, image size reduced to 46 / 2 = 23
layer0 = LeNetConvPoolLayer(
        rng1,
        input=layer0_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )
# second convolutional layer
# input image size 23X23, filter size 4X4, filter number nkerns[1]
# after filtering, image size (23 - 4 + 1) = 20
# after max pooling, image size reduced to 20 / 2 = 10    
layer1 = LeNetConvPoolLayer(
        rng2,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 23, 23),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
    
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
        rng3,
        input=layer2_input,
        n_in=nkerns[1] * 10 * 10,
        n_out=400,
        activation=T.nnet.sigmoid
    )
    
layer3 = BinaryLogisticRegression(
        rng4,
        input=layer2.output,
        n_in=400,
        n_out=basis_size * basis_size,
    )    
cost = layer3.negative_log_likelihood(y)
error = ((y - layer3.y_pred)**2).sum()

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)

updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    
#
#test_model = theano.function(
#        inputs = [index],
#        outputs = cost,
#        givens={
#            x: testInput[index * batch_size: (index + 1) * batch_size],
#            y: testOutput[index * batch_size: (index + 1) * batch_size]
#        }
#    )
    
train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x: trainInput[index : (index + 1) ],
            y: trainOutput[index : (index + 1) ]
        }
    )    

#%% training the model
    
n_train_batches = 50
n_epochs = 100
epoch = 0

while (epoch < n_epochs):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        print(('   epoch %i, minibatch %i/%i.') % (epoch, minibatch_index +1, n_train_batches))
        
#test_losses = [test_model(i) for i in range(n_test_batches)]
#test_score = np.mean(test_losses)



#%%
predict_model = theano.function(
        inputs = [x],
        outputs = layer3.p_y_given_x,
        on_unused_input='ignore'
    )

predicted_values = predict_model(testInput[5:6])


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
output_img = predicted_values
output_img = output_img.reshape(50,50)
output_img = np.asarray(output_img, dtype = 'float64') /256

plt.imshow(output_img)