# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:27:53 2018

@author: fady-
"""
import csv
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)


def f1_score(confusion_matrix, num_classes):    
    '''
    f1_score computes f1, macro_f1 scores for a confusion matrix

    returns f1, macro_f1 scores
    '''
    # get TP, TN, FT, and FN
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    # we're not using TN
#    TN = []
#    for i in range(num_classes):
#        temp = np.delete(confusion_matrix, i, 0)  # delete ith row
#        temp = np.delete(temp, i, 1)  # delete ith column
#        TN.append(sum(sum(temp)))
    # TN = np.array(TN)

    # get precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # get rid of nans if any
    precision[np.isnan(precision)] = 1
    recall[np.isnan(recall)] = 1
    # get f1 score for all classes
    f1_all_classes = np.array(2*(precision*recall)/(precision+recall))
    # get macrof1
    macro_f1 = f1_all_classes.mean()*100

    return macro_f1, f1_all_classes, precision, recall


def get_scores(session, true_labels, prediction_labels, num_classes):
    '''
    get_scores takes true and predicted labels
    
    returns f1, f1_macro, accuracy
    '''
    # load lsit batch predictions to concatenated numpy array
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
    macro_f1, f1_all_classes, precision, recall = f1_score(confusion_matrix, num_classes)
    f1_all_classes = np.array(f1_all_classes)
    f1_all_classes = np.ndarray.tolist(f1_all_classes)

    # get accuracy
    accuracy = 100 * np.sum(prediction_labels == true_labels) / len(true_labels)

    return macro_f1, f1_all_classes, accuracy


def next_batch(step, batch_size, train_x, train_y):
    '''
    next_batch takes data and labels

    returns batches for later preprocessing to train or validate
    '''
    offset = step*batch_size
    if offset+batch_size <= len(train_x):
        data = train_x[offset:offset+batch_size]
        labels = train_y[offset:offset+batch_size]
    else:
        if len(train_x[offset:])!=0:
            data = train_x[offset:]
            labels = train_y[offset:]
    return data, labels


def shuffle(X_train, y_train):
    X_train_shuffle = np.array(X_train, ndmin=2)
    order =  np.random.choice(X_train_shuffle.shape[1], X_train_shuffle.shape[1], replace=False)
    X_train_shuffle = X_train_shuffle[:, order]
    y_train_shuffle = y_train[order, :]
    X_train_shuffle = np.ndarray.tolist(X_train_shuffle)[0]
    
    return X_train_shuffle, y_train_shuffle


def plot(total_test, total_dev, total_train, batch_size, hold_prob, score, folder):
    '''
    save figures for f1, accuracy performances
    '''
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
    '''
    write accuracies throughout the whole training procedure
    '''
    scores = zip(total_test[1:], total_dev[1:], total_train[1:])
    if score=='f1':        
        with open(folder + 'macrof1_test_dev_train_b{:d}_hb{:.2f}.csv'.format(batch_size,
                  hold_prob), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('test_macroF1', 'dev_macroF1', 'train_macroF1'))
            for row in scores:
                writer.writerow(row)
    elif score=='acc':
        with open(folder + 'accuracy_test_dev_train_b{:d}_hb{:.2f}.csv'.format(batch_size, hold_prob),
                  'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('test_accuracy', 'dev_accuracy', 'train_accuracy'))
            for row in scores:
                writer.writerow(row)
 
               
def write_bestModel(test_f1_all_classes, dev_f1_all_classes, best_model_name):
    '''
    write scores for best model
    '''
    best = zip(test_f1_all_classes, dev_f1_all_classes)
    with open(best_model_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('test_F1', 'dev_F1'))
        for row in best:
            writer.writerow(row)


def save_model(session, folder, batch_size, hold_prob):
    '''
    save best model
    '''
    saver = tf.train.Saver()
    saver.save(session, folder + 'model/fine_tune_vgg16_b{:d}_hb{:.2f}'.format(
                        batch_size, hold_prob))
