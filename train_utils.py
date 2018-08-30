# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:27:53 2018

@author: Fady Baly
"""

import os
import cv2
import csv
import shutil
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)


def f1_score(confusion_matrix):
    """computes f1, macro_f1 scores for a confusion matrix
        Args:
            confusion_matrix: the confusion matrix

        returns:
            f1, macro_f1 scores
    """
    # get TP, TN, FT, and FN
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp
    # we're not using TN
    # TN = []
    # for i in range(num_classes):
    #     temp = np.delete(confusion_matrix, i, 0)  # delete ith row
    #     temp = np.delete(temp, i, 1)  # delete ith column
    #     TN.append(sum(sum(temp)))
    #  TN = np.array(TN)

    # get precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # get rid of nans if any
    precision[np.isnan(precision)] = 1
    recall[np.isnan(recall)] = 1
    # get f1 score for all classes
    f1_all_classes = np.array(2*(precision*recall)/(precision+recall))
    # get macrof1
    macro_f1 = f1_all_classes.mean()*100

    return macro_f1, f1_all_classes, precision, recall


def get_scores(session, true_labels, prediction_labels, num_classes):
    """takes true and predicted labels
        Args:
            session: the current session we're using
            true_labels: the labels provided with the dataset
            prediction_labels: the predicted labels from the model
            num_classes: the number of classes in the dataset

        Returns:
            f1, f1_macro, accuracy
    """

    # load list batch predictions to concatenated numpy array
    prediction_labels = np.vstack(prediction_labels)

    # convert from one hot encoded labels to vectorized labels
    prediction_labels = session.run(tf.argmax(prediction_labels, axis=1))
    true_labels = session.run(tf.argmax(true_labels, axis=1))

    # get the confusion matrix
    confusion_matrix = tf.confusion_matrix(true_labels, prediction_labels, num_classes=num_classes, dtype=tf.int32,
                                           name=None, weights=None)

    # get a numpy array of the confusion matrix
    confusion_matrix = session.run(confusion_matrix)

    # get macrof1 and f1 scores per class
    macro_f1, f1_all_classes, precision, recall = f1_score(confusion_matrix)
    f1_all_classes = np.array(f1_all_classes)
    f1_all_classes = np.ndarray.tolist(f1_all_classes)

    # get accuracy
    accuracy = 100 * np.sum(prediction_labels == true_labels) / len(true_labels)

    return macro_f1, f1_all_classes, accuracy


def next_batch(step, batch_size, x_data, y_label):
    """takes data and labels
        Args:
            step: the number of the batch we're in
            batch_size: the batch size for training
            x_data: the data
            y_label: the labels

        Returns:
             batches for later preprocessing to train or validate
    """
    offset = step*batch_size
    if offset+batch_size <= len(x_data):
        data = x_data[offset:offset + batch_size]
        labels = y_label[offset:offset + batch_size]
    else:
        if len(x_data[offset:]) != 0:
            data = x_data[offset:]
            labels = y_label[offset:]
    return data, labels


def shuffle(x_train, y_train):
    """shuffles the data before training
        Args:
            x_train: the training dataset
            y_train: the labels

        Returns:
            shuffled dataset
    """
    x_train_shuffle = np.array(x_train, ndmin=2)
    order = np.random.choice(x_train_shuffle.shape[1], x_train_shuffle.shape[1], replace=False)
    x_train_shuffle = x_train_shuffle[:, order]
    y_train_shuffle = y_train[order, :]
    x_train_shuffle = np.ndarray.tolist(x_train_shuffle)[0]
    
    return x_train_shuffle, y_train_shuffle


def plot(total_test, total_dev, total_train, batch_size, hold_prob, score, folder):
    """save figures for f1, accuracy performances
        Args:
            total_test: total f1/accuracy score for every epoch for test data
            total_dev: total f1/accuracy score for every epoch for dev data
            total_train: total f1/accuracy score for every epoch for train data
            folder: the directory where we want to save the model
            batch_size: the batch size of the training to included it in the model's name
            hold_prob: the dropout probability to include it in the model's name
            score: 'f1' or 'acc' to include it on the figure's name

        Returns:
            saved figures for accuracies and f1 scores during training and testing
    """
    # ignore the first element which is zero
    total_test = total_test[1:]
    total_dev = total_dev[1:]
    total_train = total_train[1:]
    # set plots, axes, title, grid and legend
    plt.plot(total_test)
    plt.plot(total_dev)
    plt.plot(total_train)
    plt.ylabel('%s score' % score)
    plt.xlabel('Epoch number')
    plt.grid(True)
    plt.title('{:s} Score for batchsize {:d} dropout {:.2f}'.format(score, batch_size, hold_prob))
    plt.legend(('test', 'dev', 'train'))

    # save fig
    plt.savefig(folder + '{:s} Score for batchsize {:d} dropout {:.2f}.png'.format(score, batch_size, hold_prob))
    plt.close()

      
def write_scores(total_test, total_dev, total_train, batch_size, folder, hold_prob, score):
    """write accuracies throughout the whole training procedure
        Args:
            total_test: total f1/accuracy score for every epoch for test data
            total_dev: total f1/accuracy score for every epoch for dev data
            total_train: total f1/accuracy score for every epoch for train data
            folder: the directory where we want to save the model
            batch_size: the batch size of the training to included it in the model's name
            hold_prob: the dropout probability to include it in the model's name
            score: 'f1' or 'acc' to include it on the figure's name

        Returns:
            excel sheet with scores for all the epochs
    """
    scores = zip(total_test[1:], total_dev[1:], total_train[1:])
    if score == 'f1':
        with open(folder + 'macrof1_test_dev_train_b{:d}_hb{:.2f}.csv'.format(batch_size,
                  hold_prob), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('test_macroF1', 'dev_macroF1', 'train_macroF1'))
            for row in scores:
                writer.writerow(row)
    elif score == 'acc':
        with open(folder + 'accuracy_test_dev_train_b{:d}_hb{:.2f}.csv'.format(batch_size, hold_prob),
                  'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('test_accuracy', 'dev_accuracy', 'train_accuracy'))
            for row in scores:
                writer.writerow(row)
 
               
def write_best_model(test_f1_all_classes, dev_f1_all_classes, best_model_name):
    """write scores for best model
        Args:
            test_f1_all_classes: the f1 score for each class in the test data
            dev_f1_all_classes: the f1 score for each class in the dev data
            best_model_name: the best model's name

        Returns:
            excel sheet with only the best scores for the best epoch
    """
    best = zip(test_f1_all_classes, dev_f1_all_classes)
    with open(best_model_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('test_F1', 'dev_F1'))
        for row in best:
            writer.writerow(row)


def save_model(session, folder, batch_size, hold_prob):
    """saves best model
        Args:
            session: the current session
            folder: the directory where we want to save the model
            batch_size: the batch size of the training to included it in the model's name
            hold_prob: the dropout probability to include it in the model's name

        Returns:
            saved model
    """
    saver = tf.train.Saver()
    saver.save(session, folder + 'model/fine_tune_vgg16_b{:d}_hb{:.2f}'.format(
                        batch_size, hold_prob))


def save_incorrect_predictions(paths, y_true, y_predicted, folder, session):
    """saves the incorrect predictions of the best trained model yet, during training.

    Args:
        paths: a list of containing all images paths
        y_true: a vector with the true labels
        y_predicted: a vector with teh predicted labels
        folder: the name of the folder in which we desire to save the wrongly predicted images

    Returns:
        saves the images in the desired directory
    """

    y_predicted = np.vstack(y_predicted)
    # convert from one hot encoded labels to vectorized labels
    y_predicted = session.run(tf.argmax(y_predicted, axis=1))
    y_true = session.run(tf.argmax(y_true, axis=1))

    matched_predictions = np.array(np.array(y_true) == np.array(y_predicted))
    wrongly_classified_indices = [i for i, e in enumerate(matched_predictions) if not e]
    wrongly_classified_pics = [paths[i] for i in wrongly_classified_indices]

    # deletes the already existed directory and create a new one
    # to save the wrong predictions of the new best model
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)

    # read the images and saves them to the new path
    for path in wrongly_classified_pics:
        img = cv2.imread(path)
        cv2.imwrite(os.path.join(folder, os.path.split(path)[1]), img)
        cv2.waitKey(0)
