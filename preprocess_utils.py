# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:07:33 2018

@author: Fady Baly
"""
import numpy as np
import cv2


global G_avg
G_avg = []
global B_avg
B_avg = []
global R_avg
R_avg = []


def pad_crop(im):
    '''
    pad_crop crops and pad images to fit the desired size
    
    returns padded cropped image
    '''
    desired_width = 224
    desired_height = 224

    im_size = im.shape  # im_size is in (height=rows, width=columns, channels=b,g,r) format
    height = im_size[0]
    width = im_size[1]
    # crop if the given is larger than the desired image size
    if width < desired_width:
        if (width / 2) % 2 != 0:
            im = im[:, :-1]
            im_size = im.shape
            height = im_size[0]
            width = im_size[1]
        pad_width = abs(int((desired_width - width) / 2))
    else:
        im = im[:, (width - desired_width) // 2:(width - desired_width) // 2 + 224, :]
        pad_width = 0

    if height < desired_height:
        if (height / 2) % 2 != 0:
            im = im[:-1, :]
            im_size = im.shape
            height = im_size[0]
            width = im_size[1]
        pad_height = abs(int((desired_height - height) / 2))
    else:
        im = im[(height - desired_height) // 2:(height - desired_height) // 2 + 224, :, :]
        pad_height = 0
    # create empty array to load the desired image size and pad the rest
    pad_im = np.zeros([desired_height, desired_width, 3])
    pad_im[pad_height:pad_height + height, pad_width:pad_width + width] = im

    return pad_im


def clr_mean(imagesNames):
    '''
    clr_mean extracts global color mean for later use
    
    returns color mean per color
    '''
    for image_name in imagesNames:
        with open(image_name, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            continue
        else:
            im = cv2.imread(image_name)
        R_avg.append(np.mean(im[0]))  # Red mean
        G_avg.append(np.mean(im[1]))  # Green mean
        B_avg.append(np.mean(im[2]))  # Blue mean

    return np.mean(np.array(R_avg)), np.mean(np.array(G_avg)), np.mean(np.array(B_avg))


def takeOutMean(dataset, B_mean, G_mean, R_mean):
    '''
    takeOutMean takes out mean from each color
    
    return mean subtracted images
    '''
    dataset[:][:, :, 0] = dataset[:][:, :, 0] - np.mean(B_mean)
    dataset[:][:, :, 1] = dataset[:][:, :, 1] - np.mean(G_mean)
    dataset[:][:, :, 2] = dataset[:][:, :, 2] - np.mean(R_mean)
    return dataset


def norm_data(dataset):
    '''
    norm_data: normalizes the given data
    
    returns normalized dataset
    '''
    x_min = 0 #dataset.min(axis=(1, 2), keepdims=True)
    x_max = 255# dataset.max(axis=(1, 2), keepdims=True)
    norm_dataset = (dataset - x_min)/(x_max - x_min)
    return norm_dataset
