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
#%%
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


#%% deep neural network
# reference: http://deeplearning.net/tutorial/lenet.html
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

rng = np.random.RandomState(23455)

# instantiate 4D tensor for input

input = T.tensor4(name = 'input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
W = theano.shared( np.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound, size = w_shp),dtype=input.dtype), name = 'W')

# initalize shared variable for bias
b_shp = (2, )
b = theano.shared(np.asarray(rng.uniform(low = -0.5, high = 0.5, size = b_shp),dtype = input.dtype), name = 'b')

#build symbolic expression that compute the convolution of input with filter W
conv_out = conv2d(input,W)
# add bias and apply activation function
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

#%% apply the convolutional layer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test.jpg')
img = np.asarray(img, dtype = 'float64') /256

img_ = img.transpose(2, 0 ,1).reshape(1, 3, 500, 422)

filtered_img = f(img_)