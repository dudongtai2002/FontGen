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
    letters = f1.readline()   #is this only read one line or the whole file?
    letters = list(letters)
    random.shuffle(letters)
    
    training_letters = letters[0:trainNum]
    testing_letters = letters[trainNum:trainNum+testNum]

    data = {'training':training_letters, 'testing':testing_letters}
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
        
        
def pp():
    print('444ss')