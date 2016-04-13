# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import os.path
import random
import numpy as np


class Font:
    
    def __init__(self, size, root_dir, input_letter, output_letter):
        self.size = size
        self.input_letter = input_letter
        self.output_letter = output_letter
        
        font_files = []
        for parent,dirnames,filenames in os.walk(root_dir):  
            for filename in filenames:
                font_files.append(os.path.join(parent,filename))
        print(('Fond %i font files') % (len(font_files)))
        random.shuffle(font_files)
        self.font_files = font_files


    def getSize(self):
        return(self.size)
        
        
    def getLetterSets(self, n_train_examples, n_test_examples):
        # return a 4D numpy array that contains images of multiple letters
        
        train_input = np.zeros((n_train_examples, len(self.input_letter),self.size,self.size))
        train_output = np.zeros((n_train_examples, len(self.output_letter),self.size,self.size))
        test_input = np.zeros((n_test_examples, len(self.input_letter),self.size,self.size))
        test_output = np.zeros((n_test_examples, len(self.output_letter),self.size,self.size))
        
        m = 0
        for font_file in self.font_files[0:n_train_examples]:
            try:
                n = 0
                for letter in self.input_letter:
                    font = ImageFont.truetype(font_file, self.size)
                    img = Image.new('L',(self.size,self.size),(1))
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0),letter,(0),font = font)
                    draw = ImageDraw.Draw(img)
                    train_input[m, n, :, :] = np.array(img)
                    n = n + 1
                    
                n = 0
                for letter in self.output_letter:
                    font = ImageFont.truetype(font_file, self.size)
                    img = Image.new('L',(self.size,self.size),(1))
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0),letter,(0),font = font)
                    draw = ImageDraw.Draw(img)
                    train_output[m, n, :, :] = np.array(img)
                    n = n + 1                                        
            except:
                continue
            m = m + 1  
        train_input = train_input[0:m,:,:,:]    
        train_output = train_output[0:m,:,:,:]   
        
        m = 0
        for font_file in self.font_files[n_train_examples:n_train_examples + n_test_examples]:
            try:
                n = 0
                for letter in self.input_letter:
                    font = ImageFont.truetype(font_file, self.size)
                    img = Image.new('L',(self.size,self.size),(1))
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0),letter,(0),font = font)
                    draw = ImageDraw.Draw(img)
                    test_input[m, n, :, :] = np.array(img)
                    n = n + 1
                n = 0
                for letter in self.output_letter:
                    font = ImageFont.truetype(font_file, self.size)
                    img = Image.new('L',(self.size,self.size),(1))
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0),letter,(0),font = font)
                    draw = ImageDraw.Draw(img)
                    test_output[m, n, :, :] = np.array(img)
                    n = n + 1                    
            except:
                continue
            m = m + 1
        i = 0  
        test_input = test_input[0:m,:,:,:]
        test_output = test_output[0:m,:,:,:]
        
        return (train_input, train_output, test_input, test_output)