# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:14:08 2018

@author: fady-
"""
import sys
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from preprocess import preprocess_batch, preprocess_validate
from train_utils import plot, get_scores, Tee, next_batch, write_scores, write_bestModel, save_model, shuffle

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

def vgg_16_training(X_train, y_train, session, last_fc, input_layer, labels_tensor, X_test, y_test,
                    X_dev, y_dev, folder, B_mean, G_mean, R_mean, batch_size=32, learn_rate=0.01,
                    num_epochs=1, num_classes=None, device_name='/CPU:0', hold_prob=0.5):
    '''
    vgg_16_training trains the model with the given parameters 
    
    saves model performances for train dev test, figures for f1 and accuracy scores, saves the best model
    '''

    log_name = folder + 'training log b{:d} hb{:.2f}'.format(batch_size, hold_prob)
    # start logging training print functions
    log = open(log_name + '.log', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    # intialize empty lists to store and write train test dev scores 
    total_f1_dev = [0]
    total_f1_train = [0]
    total_f1_test = [0]
    total_dev_accuracy = [0]
    total_train_accuracy = [0]
    total_test_accuracy = [0]
    best_epoch = 0

    # create softmax layer for predictions
    softmax_layer = tf.nn.softmax(last_fc, name='softmax')

    # design optimizer
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_fc, labels=labels_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost_function)

    num_steps = len(X_train)//batch_size + 1

    # initialize and run global variables
    init = tf.global_variables_initializer()
    session.run(init)

    # start training
    print('Let the training begin:')
    stop_counter = 0
    for epoch in range(num_epochs):
        X_train_shuffle, y_train_shuffle = shuffle(X_train, y_train)
        print('Epoch #{:d}'.format(epoch + 1))
        for step, k in zip(range(num_steps), tqdm(range(num_steps - 1))):
            # extract batches for training
            data, batch_labels = next_batch(step, batch_size, X_train_shuffle, y_train_shuffle)
            # preprocess each batch with global RGB mean extracted earlier
            batch_data = preprocess_batch(data, B_mean, G_mean, R_mean)
            # train the batch
            feed_dict = {input_layer: batch_data, labels_tensor: batch_labels}
            _, cost = session.run([optimizer, cost_function], feed_dict=feed_dict)
        # deleting batch_data to reduce memory usage
        del batch_data

        '''
        evaluate model
        '''

        # get train scores
        # create batches for predictions to avoid out of memory error
        train_predictions = []
        for step in range(len(X_train)//batch_size+1):
            data, labels = next_batch(step, batch_size, X_train, y_train)
            # get prediction per batch
            train_predictions.append(preprocess_validate(data, B_mean, G_mean, R_mean, input_layer,
                                                         labels_tensor, softmax_layer, session, labels))

        train_f1, train_f1_all_classes, train_accuracy = get_scores(session, y_train,
                                                                         train_predictions, num_classes)

        # get dev scores
        # create batches for predictions to avoid out of memory error
        dev_predictions = []
        for step in range(len(X_dev)//batch_size+1):
            data, labels = next_batch(step, batch_size, X_dev, y_dev)
            dev_predictions.append(preprocess_validate(data, B_mean, G_mean, R_mean, input_layer,
                                                       labels_tensor, softmax_layer, session, labels))

        dev_f1, dev_f1_all_classes, dev_accuracy = get_scores(session, y_dev,
                                                                   dev_predictions, num_classes)

        # get test score
        # create batches for predictions to avoid out of memory errorss
        test_predictions = []
        for step in range(len(X_test)//batch_size+1):
            data, labels = next_batch(step, batch_size, X_test, y_test)
            test_predictions.append(preprocess_validate(data, B_mean, G_mean, R_mean, input_layer,
                                                        labels_tensor, softmax_layer, session, labels))

        test_f1, test_f1_all_classes, test_accuracy = get_scores(session, y_test,
                                                                      test_predictions, num_classes)

        # print progress for each epoch
        progress = ('Epoch {:2d}/{:2d}:\n'
                    '\tTrain F1 Score = {:.2f}%\t  Dev F1 Score = {:.2f}%\t Test F1 Score = {:.2f}%\n'
                    '\tTrain Accuracy = {:.2f}%\t  Dev Accuracy = {:.2f}%\t Test Accuracy = {:.2f}%\n')
        print(progress.format(epoch + 1, num_epochs, train_f1, dev_f1, test_f1, train_accuracy,
                              dev_accuracy, test_accuracy))

        total_f1_train.append(train_f1); total_train_accuracy.append(train_accuracy)
        total_f1_dev.append(dev_f1); total_dev_accuracy.append(dev_accuracy)        
        total_f1_test.append(test_f1); total_test_accuracy.append(test_accuracy)

        # check if best f1 exists and save model
        if len(total_f1_dev) > 2:
            if total_f1_dev[-1] > max(total_f1_dev[:-1]):
                best_epoch = epoch + 1
                # counter for stopping criteria; resets if new best is found
                stop_counter = 0

                # write dev f1 scores for the best model
                best_model_name = folder + 'best model b{:d} hb{:.2f} epoch{:d}'.format(batch_size,
                                                          hold_prob, best_epoch)
                write_bestModel(test_f1_all_classes, dev_f1_all_classes, best_model_name)

                # save best dev model
                print('saving best model so far...')
                save_model(session, folder, batch_size, hold_prob)
            else:
                # counter for stopping criteria; increases when no new best is found
                stop_counter += 1
                if stop_counter > 4:
                    print('stopped at Epoch {:d}, best Epoch at {:d}'.format(
                            epoch, best_epoch) + '\n')
                    break

    # stop logging training print functions
    sys.stdout = backup
    log.close()

    # write total f1 for test dev train
    write_scores(total_f1_test, total_f1_dev, total_f1_train, batch_size,
                 folder, hold_prob, 'f1')

    # write total accuracy for test dev train
    write_scores(total_test_accuracy, total_dev_accuracy, total_train_accuracy,
                 batch_size, folder, hold_prob, 'acc')

    # plot overall performance
    plot(total_f1_test, total_f1_dev, total_f1_train, batch_size,
         hold_prob, score='F1', folder=folder)
    plot(total_test_accuracy, total_dev_accuracy, total_train_accuracy,
         batch_size, hold_prob, score='Accuracy', folder=folder)
