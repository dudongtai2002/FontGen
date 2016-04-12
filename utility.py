# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:20:00 2016

@author: shengx
"""
#%%
import json
import random
import numpy as np
import theano
from theano import tensor as T

def generateLetterSets(trainNum, testNum):
    # generate a json files that have both training and testing characters
    f1 = open('./Character/ChineseCharacter.txt','r')
    letters = f1.readline()
    letters = list(letters)
    random.shuffle(letters)
    
    training_letters = letters[0:trainNum]
    testing_letters = letters[trainNum:trainNum+testNum]

    data = {'training':training_letters, 'testing':testing_letters}
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
        
        
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