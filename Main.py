# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

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
#%%
trainInput, trainOutput, testInput, testOutput = Fonts.getLetterSets(10510,51)
trainInput = 1 - trainInput
trainOutput = 1 - trainOutput
testInput = 1 - testInput
testOutput = 1 - testOutput

n_train = trainInput.shape[0]
n_test = testInput.shape[0]
input_size = len(input_letter) * basis_size * basis_size
output_size = len(output_letter) * basis_size * basis_size
image_size = basis_size * basis_size

trainInput = trainInput.reshape((n_train,image_size*len(input_letter)))
trainOutput = trainOutput.reshape((n_train,image_size*len(output_letter)))
testInput = testInput.reshape((n_test,image_size*len(input_letter)))
testOutput = testOutput.reshape((n_test,image_size*len(output_letter)))
trainInput, trainOutput = shared_dataset(trainInput, trainOutput) 


batch_size = 50

#%% building neural networks


rng1 = np.random.RandomState(1234)
rng2 = np.random.RandomState(2345)
rng3 = np.random.RandomState(1567)
rng4 = np.random.RandomState(1124)
nkerns = [2, 2]
learning_rate = 1


# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')
y = T.imatrix('y')

print('...building the model')

layer00_input = x[:,0:image_size].reshape((batch_size, 1, basis_size, basis_size))
layer01_input = x[:,image_size:2 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer02_input = x[:,2 * image_size:3 * image_size].reshape((batch_size, 1, basis_size, basis_size))
layer03_input = x[:,3 * image_size:4 * image_size].reshape((batch_size, 1, basis_size, basis_size))
# first convolutional layer
# image original size 50X50, filter size 5X5, filter number nkerns[0]
# after filtering, image size reduced to (36 - 3 + 1) = 34
# after max pooling, image size reduced to 34 / 2 = 17
layer00 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer01 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer02 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )
layer03 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03_input,
        image_shape=(batch_size, 1, basis_size, basis_size),   # input image shape
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )    

        
# second convolutional layer
# input image size 23X23, filter size 4X4, filter number nkerns[1]
# after filtering, image size (17 - 4 + 1) = 14
# after max pooling, image size reduced to 14 / 2 = 7    
layer10 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer11 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer12 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer02.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )
layer13 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer03.output,
        image_shape=(batch_size, nkerns[0], 17, 17),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2)
    )    
    
# layer 2 input size = 2 * 4 * 7 * 7 =  392   
layer2_input = T.concatenate([layer10.output.flatten(2), layer11.output.flatten(2), layer12.output.flatten(2), layer13.output.flatten(2)],
                              axis = 1)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2_input,
        n_in=nkerns[1] * len(input_letter) * 7 * 7,
        n_out=50,
        activation=T.nnet.sigmoid
    )
layer3 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2.output,
        n_in=50,
        n_out=50,
        activation=T.nnet.sigmoid
    )
    
layer4 = BinaryLogisticRegression(
        np.random.RandomState(np.random.randint(10000)),
        input=layer3.output,
        n_in=50,
        n_out=basis_size * basis_size,
    )    
cost = layer4.negative_log_likelihood(y)
error = ((y - layer4.y_pred)**2).sum()

params = (layer4.params 
        + layer3.params 
        + layer2.params 
        + layer10.params + layer11.params + layer12.params + layer13.params
        + layer00.params + layer01.params + layer02.params + layer03.params)
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
            x: trainInput[index * batch_size: (index + 1) * batch_size],
            y: trainOutput[index * batch_size: (index + 1) * batch_size]
        }
    )    

#%% training the model
    
n_train_batches = 210
n_epochs = 1500
epoch = 0

while (epoch < n_epochs):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        print(('   epoch %i, minibatch %i/%i.') % (epoch, minibatch_index +1, n_train_batches))
        
#test_losses = [test_model(i) for i in range(n_test_batches)]
#test_score = np.mean(test_losses)



#%% predict output


predict_model = theano.function(
        inputs = [x],
        outputs = layer4.p_y_given_x,
        on_unused_input='ignore'
    )

predicted_values = predict_model(testInput[0:50])

#%% compare output
n = 3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
output_img = predicted_values
output_img = output_img.reshape(50,36,36)
output_img = np.asarray(output_img, dtype = 'float64') /256
plt.figure(1)
plt.subplot(121)
plt.imshow(output_img[n,:,:],interpolation="nearest",cmap='Greys')
plt.subplot(122)
plt.imshow(testOutput[n,:].reshape((basis_size,basis_size)),interpolation="nearest",cmap='Greys')
