# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:43:57 2016

@author: shengx
"""

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json


class Font:
    
    def __init__(self, size, fontFile):
        self.size = size
        filename = './Fonts/'+fontFile
        self.font = ImageFont.truetype(filename, size)

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
        
    def getLetterSets(self):
        # return a 3D numpy array that contains images of multiple letters
        with open('data.json') as json_data:
            data = json.load(json_data)    
            
        training_letter = data['training']
        testing_letter = data['testing']
        
        training = np.zeros((self.size,self.size,len(training_letter)))
        testing = np.zeros((self.size,self.size,len(testing_letter)))
        
        i = 0
        for letter in training_letter:
            img = Image.new('L',(self.size,self.size),(1))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0),letter,0,font=self.font)
            draw = ImageDraw.Draw(img)
            training[:,:,i] = np.array(img)
            i = i+1
        i = 0    
        for letter in testing_letter:
            img = Image.new('L',(self.size,self.size),(1))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0),letter,0,font=self.font)
            draw = ImageDraw.Draw(img)
            testing[:,:,i] = np.array(img)
            i = i+1
                        
        return (training, testing)
        
        

    
    
