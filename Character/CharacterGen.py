# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:11:18 2016

@author: shengx
"""
''' 
editing common Chinese Characters
This will result in a file that only have one line of common Chinese Characters
'''
f1 = open('现代汉语常用字表.txt','r')
f2 = open('ChineseCharacter.txt','w')
for line in f1:
    line = line.rstrip()
    f2.write(line.replace(' ',''))
f1.close()
f2.close()

