# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

#%%

from Font import *
from utility import *
import tensorflow as tf

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


#%% deep neural network
# reference: http://deeplearning.net/tutorial/lenet.html

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
