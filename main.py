# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:33:40 2018

@author: Fady Baly
"""
import os
import time
import argparse
import tensorflow as tf
from models import vgg16
from keras.utils import np_utils
from train import vgg_16_training
from preprocess import getImageNames
from preprocess_utils import clr_mean
from sklearn.model_selection import train_test_split

def main(arguments):
    max_num_epochs = arguments.me
    batch_size = arguments.batchsize
    hold_prob = arguments.holdprobabilty
    numberofimages = arguments.numberOfimages
    cd = arguments.choosedevice
    dataset_folder = arguments.foldername

    device_name = ['/CPU:0', '/GPU:0']
    if device_name[cd] == '/CPU:0':        
        print('Using CPU')
    else:
        print('Using GPU')

    # extracting images names and get mean per color
    imagesNames, labels = getImageNames(number_of_images = numberofimages, orig_data=dataset_folder)
    R_mean, G_mean, B_mean = clr_mean(imagesNames)

    # split data to test and train_dev
    x_train_dev, x_test, y_train_dev, y_test = train_test_split(imagesNames, labels, test_size=0.1,
                                                                random_state=17)
    assert len(set(y_test)) == len(set(y_train_dev))
    # split train_dev to train and dev
    x_train, x_dev, y_train, y_dev = train_test_split(x_train_dev, y_train_dev, test_size=0.1,
                                                      random_state=17)
    assert len(set(y_dev)) == len(set(y_train)) == len(set(y_test))

    # one hot encode the labels
    num_classes = len(set(y_test))
#    print(np.array(y_train).shape, np.array(y_dev).shape, np.array(y_test).shape)
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_dev = np_utils.to_categorical(y_dev, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    folder = '{:d} classes results batchsize {:d} holdprob {:.2f}/'.format(num_classes, batch_size, hold_prob)
    # create folders to save results and model
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + 'model/'):
        os.makedirs(folder + 'model/')

    '''
    initialize model
    '''

    # better allocate memory during training in GPU
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    # create input and label tensors placeholder
    with tf.device(device_name[cd]):
        input_layer = tf.placeholder(tf.float32, [None, 224, 224, 3])
        labels_tensor = tf.placeholder(tf.float32, [None, num_classes])

    # start tensorflow session
    with tf.Session(config=config) as sess:
        # create the vgg16 model
        tic = time.clock()
        vgg16_model = vgg16(input_layer, 'model_weights/vgg16_weights.npz', sess, hold_prob=hold_prob,
                            num_classes=num_classes, device_name=device_name[cd])
        toc = time.clock()
        print('loading model time: ', toc-tic)

        # train and test vgg16
        vgg_16_training(X_train=x_train, y_train=y_train, session=sess, last_fc=vgg16_model.fc3l,
                        input_layer=input_layer, labels_tensor=labels_tensor, X_test=x_test, y_test=y_test,
                        folder=folder, batch_size=batch_size, learn_rate=1e-4, num_epochs=max_num_epochs,
                        num_classes=num_classes, device_name=device_name[cd], X_dev=x_dev, y_dev=y_dev,
                        hold_prob=hold_prob, B_mean = B_mean, G_mean=G_mean, R_mean=R_mean)

if __name__ == "__main__":
    # get_available_processors()
    parser = argparse.ArgumentParser()
    parser.add_argument('-me', help='maximum number of epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', help="provide list of desired batch sizes", type=int, default=64)
    parser.add_argument('-hb', "--holdprobabilty", help='the desired hold probability (max is 0, min is 1)',
                        type=float, default=0.75)
    parser.add_argument('-nm', '--numberOfimages', help='provide the number of images you want to train on, leave '
                                                        'empty if you want all the data', default=None)
    parser.add_argument('-cd', '--choosedevice', help='pass 0 CPU or leave empty for gpu', type=int, default=1)
    parser.add_argument('-f', '--foldername', help='desired dataset folder name', type=str, default='../dermnet dataset/')
    args = parser.parse_args()

    main(args)
