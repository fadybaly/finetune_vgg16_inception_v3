# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:21:18 2018

@author: fady-
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 20:37:51 2018

@author: Fady Baly
"""

import os
import cv2
import glob
import numpy as np
from preprocess_utils import pad_crop, norm_data


def preprocess_batch(imagesNames, B_mean, G_mean, R_mean):
    '''
    preprocess_batch preprocess batches with pre-computed color means
    to be fed during training on the go with each iteration
    
    return preprocessed 4-D array for given images names
    '''
    # load, remove mean, crop images
    dataset = []
    for image_name in imagesNames:
#        with open(image_name, 'rb') as f:
#            check_chars = f.read()[-2:]
#        if check_chars != b'\xff\xd9':
#            continue
#        else:
#            im = cv2.imread(image_name)
        im = cv2.imread(image_name)
        im[0] = im[0] - R_mean
        im[1] = im[1] - G_mean
        im[2] = im[2] - B_mean
        pad_crop_im = pad_crop(im)
        pad_crop_BGR_im = cv2.cvtColor(np.uint8(pad_crop_im), cv2.COLOR_RGB2BGR)
        dataset.append(pad_crop_BGR_im)
    # convert list of images to 4-D array
    dataset = np.array(dataset)
    # normalize iamges
    dataset = norm_data(dataset)
    return dataset

def preprocess_validate(imagesNames, B_mean, G_mean, R_mean, input_layer, labels_tensor, softmax_layer, session, y_true):
    '''
    preprocess_validate preprocess batches with pre-computed color means
    to be fed for model validation
    
    returns predictions for the given images names 
    '''
    # load, remove mean, crop images      
    dataset = []
    for image_name in imagesNames:
#        with open(image_name, 'rb') as f:
#            check_chars = f.read()[-2:]
#        if check_chars != b'\xff\xd9':
#            continue
#        else:
#            im = cv2.imread(image_name)
        im = cv2.imread(image_name)
        im[0] = im[0] - R_mean
        im[1] = im[1] - G_mean
        im[2] = im[2] - B_mean
        pad_crop_im = pad_crop(im)
        pad_crop_BGR_im = cv2.cvtColor(np.uint8(pad_crop_im), cv2.COLOR_RGB2BGR)
        dataset.append(pad_crop_BGR_im)
    # convert list of images to 4-D array
    dataset = np.array(dataset)
    # normalize iamges
    dataset = norm_data(dataset)
    # predict labels for provided images names
    predictions = session.run(softmax_layer, feed_dict={input_layer: dataset, labels_tensor: y_true})

    return predictions


def getImageNames(number_of_images=None, orig_data='../dermnet dataset/'):
    '''
    getImageNames get images names to be later preprocessed during training and validation
    
    returns images names and the labels for each image
    '''
    # pass the directory which contains the dataset
    directory = orig_data
    directories = os.listdir(directory)
    labels = []
    images_names = []
    i = 1
    label = -1
    
    # labeling each image per directory
    for folder in directories:
        if i == number_of_images:
            break
        label += 1
        # get images names
        for image_name in glob.glob(directory + folder + '/*.jpg'):
            with open(image_name, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                continue
            else:                
                if i == number_of_images:
                    break
                labels.append(label)
                images_names.append(image_name)

    return images_names, labels