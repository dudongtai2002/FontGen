# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:43:57 2016

@author: shengx
"""
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class Font:
    
    def __init__(self, size, fontFile):
        self.size = size
        self.font = ImageFont.truetype('./Font/'+fontFile, size)
        
    def getSize(self):
        return(self.size)
        
    def getSingleLetter(self,letter):
        # returning a 2D numpy array of the specified letter-font image
        img = Image.new('L',(self.size,self.size),(1))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0),letter,(0,0,0),font=self.font)
        draw = ImageDraw.Draw(img)
        arr = np.array(img)
        return(arr)
        
    def showSingleLetter(self,letter):
        img = Image.new('L',(self.size,self.size),(1))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0),letter,(0,0,0),font=self.font)
        draw = ImageDraw.Draw(img)
        arr = np.array(img)
        plt.imshow(arr)
        plt.show()
        
    def getLetterSets(self,file):
        # return a 3D numpy array that contains images of multiple letters
        f1 = open(file,'r')
        line = f1.readline()
        
        letters = np.zeros((self.size,self.size,len(line)));
        i = 0
        for letter in line:
            img = Image.new('L',(self.size,self.size),(1))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0),letter,(0,0,0),font=font)
            draw = ImageDraw.Draw(img)
            letters[:,:,i] = np.array(img)
            i = i+1
        return letters 
        
        