# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:33:40 2018

@author: Fady Baly
"""

import os
import time
import argparse
import tensorflow as tf
from keras.utils import np_utils
from train import start_training
from preprocess_utils import clr_mean
from models import Vgg16, InceptionV3
from preprocess import get_images_names
from sklearn.model_selection import train_test_split


def main(arguments):
    x_data = dict()
    y_data = dict()
    tensors = dict()
    color_data = dict()
    flags = dict()
    flags['num_epochs'] = arguments.me
    flags['batch_size'] = arguments.batchsize
    flags['hold_prob'] = arguments.holdprobabilty
    flags['numberofimages'] = arguments.numberOfimages
    flags['cd'] = arguments.choosedevice
    flags['dataset_folder'] = arguments.foldername
    flags['model'] = arguments.model
    flags['learn_rate'] = 1e-4
    flags['vgg16_weights'] = 'model_weights/vgg16_weights.npz'
    flags['inception_v3_weights'] = 'model_weights/inception_v3.ckpt'
    flags['chatbot_tensor'] = []

    device_name = ['/CPU:0', '/GPU:0']
    if device_name[flags['cd']] == '/CPU:0':
        print('Using CPU')
    else:
        print('Using GPU')

    # extracting images names and get mean per color
    images_names, labels = get_images_names(number_of_images=flags['numberofimages'],
                                            orig_data=flags['dataset_folder'])
    color_data['r_mean'], color_data['g_mean'], color_data['b_mean'] = clr_mean(images_names)

    # split data to test and train_dev
    x_train_dev, x_data['x_test'], y_train_dev, y_data['y_test'] = train_test_split(images_names,
                                                                                    labels,
                                                                                    test_size=0.1,
                                                                                    random_state=17)
    assert len(set(y_data['y_test'])) == len(set(y_train_dev))

    # split train_dev to train and dev
    x_data['x_train'], x_data['x_dev'], y_data['y_train'], y_data['y_dev'] = train_test_split(x_train_dev,
                                                                                              y_train_dev,
                                                                                              test_size=0.1,
                                                                                              random_state=17)
    assert len(set(y_data['y_dev'])) == len(set(y_data['y_train'])) == len(set(y_data['y_test']))

    # one hot encode the labels
    flags['num_classes'] = len(set(y_data['y_test']))
    # print(np.array(y_train).shape, np.array(y_dev).shape, np.array(y_test).shape)
    y_data['y_train'] = np_utils.to_categorical(y_data['y_train'], flags['num_classes'])
    y_data['y_dev'] = np_utils.to_categorical(y_data['y_dev'], flags['num_classes'])
    y_data['y_test'] = np_utils.to_categorical(y_data['y_test'], flags['num_classes'])

    flags['folder'] = '{:d} classes results batchsize {:d} holdprob {:.2f}/'.format(flags['num_classes'],
                                                                                    flags['batch_size'],
                                                                                    flags['hold_prob'])
    # create folders to save results and model
    if not os.path.exists(flags['folder']):
        os.makedirs(flags['folder'])
    if not os.path.exists(flags['folder'] + 'model/'):
        os.makedirs(flags['folder'] + 'model/')

    '''
    initialize model
    '''

    # better allocate memory during training in GPU
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    # create input and label tensors placeholder
    with tf.device(device_name[flags['cd']]):
        tensors['hold_prob'] = tf.placeholder_with_default(1, shape=(), name='hold_prob')
        if flags['model'] == 'vgg16':
            tensors['input_layer'] = tf.placeholder(tf.float32, [None, 224, 224, 3], 'input_layer')
        else:
            tensors['input_layer'] = tf.placeholder(tf.float32, [None, 299, 299, 3], 'input_layer')
        tensors['labels_tensor'] = tf.placeholder(tf.float32, [None, flags['num_classes']])
        if flags['chatbot_tensor']:
            tensors['chatbot_tensor'] = tf.placeholder(tf.float32, [None, 10], 'chatbot_tensor')

    # start tensorflow session
    with tf.Session(config=config) as sess:
        # create the vgg16 model
        tic = time.clock()
        if flags['model'] == 'vgg16':
            model = Vgg16(tensors['input_layer'], tensors['chatbot_tensor'], flags['vgg16_weights'], sess,
                          hold_prob=tensors['hold_prob'],
                          num_classes=flags['num_classes'])
        else:
            model = InceptionV3(tensors['input_layer'], flags['inception_v3_weights'], sess,
                                flags['hold_prob'],
                                num_classes=flags['num_classes'])
        toc = time.clock()
        print('loading model time: ', toc-tic)

        writer = tf.summary.FileWriter('tensorboard')
        writer.add_graph(sess.graph)
        print('start tensorboard')

        # train, dev, and test
        start_training(x_data=x_data, y_data=y_data, flags=flags, color_data=color_data, session=sess,
                       tensors=tensors, last_fc=model.last_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-me', help='maximum number of epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', help="provide list of desired batch sizes", type=int, default=64)
    parser.add_argument('-hb', '--holdprobabilty', help='the desired hold probability (max is 0, min is 1)',
                        type=float, default=0.75)
    parser.add_argument('-nm', '--numberOfimages', help='provide the number of images you want to train on, leave '
                                                        'empty if you want all the data', default=None)
    parser.add_argument('-cd', '--choosedevice', help='pass 0 CPU or leave empty for gpu', type=int, default=1)
    parser.add_argument('-f', '--foldername', help='desired dataset folder name', type=str,
                        default='../dermnet dataset/')
    parser.add_argument('-m', '--model', help='choose "inception_v3" or "vgg16"', type=str,
                        default='vgg16')
    args = parser.parse_args()
    main(args)
