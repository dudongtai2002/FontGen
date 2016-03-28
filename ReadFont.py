# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:26:06 2016

@author: shengx
"""

#%%
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#%%
font = ImageFont.truetype('./Font/simkai.ttf',100)
img = Image.new('L',(100,100),(1))
draw = ImageDraw.Draw(img)
draw.text((0, 0),"翻",(0,0,0),font=font)
draw = ImageDraw.Draw(img)
arr = np.array(img)
imgplot = plt.imshow(arr)
#%%
f1 = open('./Character/ChineseCharacter.txt','r')
line = f1.readline()

letters = np.zeros((100,100,3500));

for letter in line:
    