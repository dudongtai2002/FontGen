# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

This file is just for general testing
"""

#%%

from Font import *
from utility import *


generateLetterSets(1000, 50)

inputLetter = Font(100,'simsun.ttf') 

trainInput, testInput = inputLetter.getLetterSets()

outputLetter = Font(100,'msyhbd.ttf') 

trainOutput, testOutput = outputLetter.getLetterSets()