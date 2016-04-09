# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

#%%

from Font import *
from utility import *

basis_size = 50
training_size = 1000
testing_size = 50
generateLetterSets(training_size, testing_size)

inputFont = Font(basis_size,'simsun.ttf') 

trainInput, testInput = inputFont.getLetterSets()

outputFont = Font(basis_size,'msyhbd.ttf') 

trainOutput, testOutput = outputFont.getLetterSets()


trainInput = np.reshape(trainInput, (basis_size**2 , training_size))
testInput = np.reshape(testInput, (basis_size**2, testing_size))
trainOutput = np.reshape(trainOutput, (basis_size**2, training_size))
testOutput = np.reshape(testOutput, (basis_size**2, testing_size))

trainInput = np.transpose(trainInput)
testInput = np.transpose(testInput)
trainOutput = np.transpose(trainOutput)
testOutput = np.transpose(testOutput)

#%%
from Font import *
from utility import *
import numpy as np

with np.load('train_data.npz') as data:
    trainInput = data['trainInput']
    trainOutput = data['trainOutput']
    testInput = data['testInput']
    testOutput = data['testOutput']
    basis_size = int(data['basis_size'])
    testing_size = int(data['testing_size'])
    training_size = int(data['training_size'])


#%% convert data to theano format 
trainInput = np.transpose(trainInput)
testInput = np.transpose(testInput)
trainOutput = np.transpose(trainOutput)
testOutput = np.transpose(testOutput)  # 1st dimension is the number of examples, 2nd dimension is vectorized images

test_set = (testInput, testOutput)   
train_set = (trainInput, trainOutput)  # a tuple for input and output

rval = [test_set, train_set]  # a list for both training and testing, with the first element being testing and second element being training

# compute minibatches
batch_size = 50
n_train_batches = trainInput.shape[0] // batch_size
n_test_batches = testOutput.shape[0] // batch_size



#%% building the actual model
from NN import *
print('...building the model')

index = T.lscalar()
x = T.matrix('x')
y = T.matrix('y')

rng = np.random.RandomState(1234)
# construct a MLP class
n_hidden = 100
L1_reg = 0.05
L2_reg = 0
learning_rate = 0.2
classifier = MLP(rng = rng, input = x, n_in = basis_size**2, n_hidden=n_hidden, n_out = basis_size**2)
cost = (
        classifier.negative_log_likelihood(y) 
        + L1_reg * classifier.L1 
        + L2_reg * classifier.L2_sqr
        )
        
test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: testInput[index * batch_size: (index + 1) * batch_size],
            y: testOutput[index * batch_size: (index + 1) * batch_size]
        }
    )

gparams = [T.grad(cost, param) for param in classifier.params]

updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: trainInput[index * batch_size: (index + 1) * batch_size],
            y: trainOutput[index * batch_size: (index + 1) * batch_size]
        }
    )    
#%% training the model
    
epoch = 0
while (epoch < 10) :
    epoch = epoch +1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        print(('   epoch %i, minibatch %i/%i.') % (epoch, minibatch_index +1, n_train_batches))
        
test_losses = [test_model(i) for i in range(n_test_batches)]
test_score = np.mean(test_losses)









#%% apply the convolutional layer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test.jpg')
img = np.asarray(img, dtype = 'float64') /256

img_ = img.transpose(2, 0 ,1).reshape(1, 3, 500, 422)

filtered_img = f(img_)