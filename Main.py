# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx

Main function
"""

#%%

from Font import *
from utility import *


generateLetterSets(1000, 50)

inputFont = Font(100,'simsun.ttf') 

trainInput, testInput = inputFont.getLetterSets()

outputFont = Font(100,'msyhbd.ttf') 

trainOutput, testOutput = outputFont.getLetterSets()