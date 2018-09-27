# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:14:08 2018

@author: Fady Baly
"""

import sys
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from preprocess import preprocess_batch, preprocess_validate
from train_utils import plot, get_scores, Tee, next_batch,\
    write_scores, write_best_model, save_model, shuffle, save_incorrect_predictions

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


def start_training(x_data, y_data, flags, color_data, session, tensors, last_fc):
    """Trains the model with the given parameters
    Args:
        x_data: train, dev, test data
        y_data: train, dev, test labels
        flags: training parameters
        color_data: RGB color means
        tensors: input and label tensors
        last_fc: logits layer
        session: the current working session

    Returns:
        saves model performances for train dev test, figures for f1 and accuracy scores, saves the best model
    """

    log_name = flags['folder'] + 'training log b{:d} hb{:.2f}'.format(flags['batch_size'], flags['hold_prob'])
    # start logging training print functions
    log = open(log_name + '.log', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    # initialize empty lists to store and write train test dev scores
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
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last_fc,
                                                                           labels=tensors['labels_tensor']))
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['learn_rate']).minimize(cost_function)

    num_steps = len(x_data['x_train']) // flags['batch_size'] + 1

    # initialize and run global variables
    init = tf.global_variables_initializer()
    session.run(init)

    # start training
    print('Let the training begin:')
    stop_counter = 0
    for epoch in range(flags['num_epochs']):
        x_train_shuffle, y_train_shuffle = shuffle(x_data['x_train'], y_data['y_train'])
        print('Epoch #{:d}'.format(epoch + 1))
        for step, k in zip(range(num_steps), tqdm(range(num_steps - 1))):
            # extract batches for training
            data, batch_labels = next_batch(step, flags['batch_size'], x_train_shuffle, y_train_shuffle)
            # preprocess each batch with global RGB mean extracted earlier
            batch_data = preprocess_batch(data, color_data['b_mean'], color_data['g_mean'], color_data['r_mean'],
                                          flags['model'])
            # train the batch
            feed_dict = {tensors['input_layer']: batch_data, tensors['labels_tensor']: batch_labels,
                         tensors['hold_prob']: flags['hold_prob']}
            _, cost = session.run([optimizer, cost_function], feed_dict=feed_dict)
            # deleting batch_data to reduce memory usage
            del batch_data

        '''
        evaluate model
        '''

        # get train scores
        # create batches for predictions to avoid out of memory error
        train_predictions = []
        for step in range(len(x_data['x_train']) // flags['batch_size'] + 1):
            data, labels = next_batch(step, flags['batch_size'], x_data['x_train'], y_data['y_train'])

            # get prediction per batch
            if data is not None:
                train_predictions.append(preprocess_validate(data, color_data, tensors, softmax_layer, session,
                                                             labels, flags['model']))

        train_f1, train_f1_all_classes, train_accuracy = get_scores(session, y_data['y_train'],
                                                                    train_predictions, flags['num_classes'])

        # get dev scores
        # create batches for predictions to avoid out of memory error
        dev_predictions = []
        for step in range(len(x_data['x_dev']) // flags['batch_size'] + 1):
            data, labels = next_batch(step, flags['batch_size'], x_data['x_dev'], y_data['y_dev'])
            if data is not None:
                dev_predictions.append(preprocess_validate(data, color_data, tensors, softmax_layer, session,
                                                           labels, flags['model']))

        dev_f1, dev_f1_all_classes, dev_accuracy = get_scores(session, y_data['y_dev'],
                                                              dev_predictions, flags['num_classes'])

        # get test score
        # create batches for predictions to avoid out of memory errors
        test_predictions = []
        for step in range(len(x_data['x_test']) // flags['batch_size'] + 1):
            data, labels = next_batch(step, flags['batch_size'], x_data['x_test'], y_data['y_test'])
            if data is not None:
                test_predictions.append(preprocess_validate(data, color_data, tensors, softmax_layer, session,
                                                            labels, flags['model']))

        test_f1, test_f1_all_classes, test_accuracy = get_scores(session, y_data['y_test'],
                                                                 test_predictions, flags['num_classes'])

        # print progress for each epoch
        progress = ('Epoch {:2d}/{:2d}:\n'
                    '\tTrain F1 Score = {:.2f}%\t  Dev F1 Score = {:.2f}%\t Test F1 Score = {:.2f}%\n'
                    '\tTrain Accuracy = {:.2f}%\t  Dev Accuracy = {:.2f}%\t Test Accuracy = {:.2f}%\n')
        print(progress.format(epoch + 1, flags['num_epochs'], train_f1, dev_f1, test_f1, train_accuracy,
                              dev_accuracy, test_accuracy))

        total_f1_train.append(train_f1)
        total_train_accuracy.append(train_accuracy)
        total_f1_dev.append(dev_f1)
        total_dev_accuracy.append(dev_accuracy)
        total_f1_test.append(test_f1)
        total_test_accuracy.append(test_accuracy)

        # check if best f1 exists and save model
        if len(total_f1_dev) > 2:
            if total_f1_dev[-1] > max(total_f1_dev[:-1]):
                best_epoch = epoch + 1
                # counter for stopping criteria; resets if new best is found
                stop_counter = 0

                # write dev f1 scores for the best model
                best_model_name = flags['folder'] + 'best model b{:d} hb{:.2f} epoch{:d}'.format(flags['batch_size'],
                                                                                                 flags['hold_prob'],
                                                                                                 best_epoch)
                write_best_model(test_f1_all_classes, dev_f1_all_classes, best_model_name)

                # save incorrect dev and test predictions for the best model
                save_incorrect_predictions(x_data['x_dev'], y_data['y_dev'], dev_predictions,
                                           flags['folder'] + 'misclassified_dev/', session)
                save_incorrect_predictions(x_data['x_test'], y_data['y_test'], test_predictions,
                                           flags['folder'] + 'misclassified_test/', session)

                # save best dev model
                print('saving best model so far...')
                save_model(session, flags)
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
    write_scores(total_f1_test, total_f1_dev, total_f1_train, flags, 'f1')

    # write total accuracy for test dev train
    write_scores(total_test_accuracy, total_dev_accuracy, total_train_accuracy, flags, 'acc')

    # plot overall performance
    # plot(total_f1_test, total_f1_dev, total_f1_train, flags, 'F1')
    # plot(total_test_accuracy, total_dev_accuracy, total_train_accuracy, flags, score='Accuracy')
