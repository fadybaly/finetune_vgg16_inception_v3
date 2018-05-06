# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:27:15 2018

@author: fady-
"""

import cv2
from preprocess import getImageNames
im = []
imagesNames, labels = getImageNames(number_of_images = None, orig_data='../test/')

for name in imagesNames:
    with open(name, 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        print('Not complete image')
    else:
        im.append(cv2.imread(name))