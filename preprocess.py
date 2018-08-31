# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:21:18 2018

@author: Fady Baly
"""

import os
import cv2
import glob
import numpy as np
from preprocess_utils import pad_crop, norm_data


def preprocess_batch(images_names, b_mean, g_mean, r_mean, model):
    """preprocess_batch preprocess batches with pre-computed color means
        to be fed during training on the go with each iteration

        Args:
            images_names: list of the batch paths
            b_mean: global blue color mean
            g_mean: global green color mean
            r_mean: global red color mean
            model: define model as inception_v3 or vgg16

        Returns:
             preprocessed 4-D array for given images names
    """
    # load, remove mean, crop images
    dataset = []
    for image_name in images_names:
        # with open(image_name, 'rb') as f:
        #     check_chars = f.read()[-2:]
        # if check_chars != b'\xff\xd9':
        #     continue
        # else:
        #     im = cv2.imread(image_name)
        im = cv2.imread(image_name)
        im[0] = im[0] - r_mean
        im[1] = im[1] - g_mean
        im[2] = im[2] - b_mean
        pad_crop_im = pad_crop(im, model)
        pad_crop_bgr_im = cv2.cvtColor(np.uint8(pad_crop_im), cv2.COLOR_RGB2BGR)
        dataset.append(pad_crop_bgr_im)
    # convert list of images to 4-D array
    dataset = np.array(dataset)
    # normalize images
    dataset = norm_data(dataset)
    return dataset


def preprocess_validate(images_names, color_data, tensors, softmax_layer,
                        session, y_true, model):
    """preprocess_validate preprocess batches with pre-computed color means
        to be fed for model validation

        Args:
            images_names: paths of images
            color_data: RGB color mean
            tensors: input label tensors
            softmax_layer: softmax layer of the model
            session: the current working session
            y_true: the true labels of the data
            model: models name (vgg16 or inception_v3)
    
        Returns:
             predictions for the given images names
    """
    # load, remove mean, crop images      
    dataset = []
    for image_name in images_names:
        # with open(image_name, 'rb') as f:
        #     check_chars = f.read()[-2:]
        # if check_chars != b'\xff\xd9':
        #     continue
        # else:
        #     im = cv2.imread(image_name)
        im = cv2.imread(image_name)
        im[0] = im[0] - color_data['r_mean']
        im[1] = im[1] - color_data['g_mean']
        im[2] = im[2] - color_data['b_mean']
        pad_crop_im = pad_crop(im, model=model)
        pad_crop_bgr_im = cv2.cvtColor(np.uint8(pad_crop_im), cv2.COLOR_RGB2BGR)
        dataset.append(pad_crop_bgr_im)
    # convert list of images to 4-D array
    dataset = np.array(dataset)
    # normalize images
    dataset = norm_data(dataset)
    # predict labels for provided images names
    predictions = session.run(softmax_layer, feed_dict={tensors['input_layer']: dataset,
                                                        tensors['labels_tensor']: y_true})

    return predictions


def get_images_names(number_of_images=None, orig_data='../dermnet dataset/'):
    """get images names to be later preprocessed during training and validation

        Args:
            number_of_images: the desired number of images if wanted to test on smaller dataset
            orig_data: the path to the dataset
        Returns:
             images names and the labels for each image
    """
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
