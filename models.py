# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:18:49 2018

@author: Fady Baly
"""

import tensorflow as tf
import numpy as np
import os

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


class Vgg16:
    def __init__(self, imgs, weights, sess, hold_prob, num_classes):
        # with tf.device(device_name):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers(hold_prob, num_classes)
        # self.softmax = tf.nn.softmax(self.fc3l)
        self.last_layer = self.fc3l
        sess.run(tf.global_variables_initializer())

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([0, 0, 0], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            # pool1
            self.pool1 = tf.nn.max_pool(self.conv1_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            # pool2
            self.pool2 = tf.nn.max_pool(self.conv2_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            # pool3
            self.pool3 = tf.nn.max_pool(self.conv3_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            # pool4
            self.pool4 = tf.nn.max_pool(self.conv4_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=True, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

            # pool5
            self.pool5 = tf.nn.max_pool(self.conv5_3,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='pool4')

    def fc_layers(self, hold_prob, num_classes):

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32,
                                                   stddev=1e-3), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l, name=scope)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32,
                                                   stddev=1e-3), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l, name=scope)
            self.parameters += [fc2w, fc2b]

        # Dropout
        with tf.name_scope('dropout') as scope:
            self.dropout = tf.nn.dropout(self.fc2, keep_prob=hold_prob)

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([1000, num_classes], dtype=tf.float32,
                                                   stddev=1e-3), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.dropout, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        print('loading weights')
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
            if i == 26:
                break


class InceptionV3:
    def __init__(self, imgs, checkpoint_path, sess, dropout_keep_prob, num_classes):
        self.end_points = {}
        self.variables_to_restore = []
        self.imgs = imgs
        self.inception_v3_base(imgs,
                               min_depth=16,
                               depth_multiplier=1.0,
                               scope='InceptionV3')

        self.inception_v3(imgs,
                          num_classes=num_classes,
                          is_training=True,
                          dropout_keep_prob=dropout_keep_prob,
                          min_depth=16,
                          depth_multiplier=1.0,
                          prediction_fn=slim.softmax,
                          spatial_squeeze=True,
                          reuse=None,
                          create_aux_logits=True,
                          scope='InceptionV3',
                          global_pool=False)
        self.last_layer = self.end_points['Logits']
        self.load_weights(checkpoint_path, sess)

    def inception_v3_base(self, inputs,
                          min_depth=16,
                          depth_multiplier=1.0,
                          scope=None):

        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        with tf.variable_scope(scope, 'InceptionV3', [inputs]):
            # conv_layers
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='VALID'):
                # 299 x 299 x 3
                end_point = 'Conv2d_1a_3x3'
                self.net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
                self.end_points[end_point] = self.net

                # 149 x 149 x 32
                end_point = 'Conv2d_2a_3x3'
                self.net = slim.conv2d(self.net, depth(32), [3, 3], scope=end_point)
                self.end_points[end_point] = self.net

                # 147 x 147 x 32
                end_point = 'Conv2d_2b_3x3'
                self.net = slim.conv2d(self.net, depth(64), [3, 3], padding='SAME', scope=end_point)
                self.end_points[end_point] = self.net

                # 147 x 147 x 64
                end_point = 'MaxPool_3a_3x3'
                self.net = slim.max_pool2d(self.net, [3, 3], stride=2, scope=end_point)
                self.end_points[end_point] = self.net

                # 73 x 73 x 64
                end_point = 'Conv2d_3b_1x1'
                self.net = slim.conv2d(self.net, depth(80), [1, 1], scope=end_point)
                self.end_points[end_point] = self.net

                # 73 x 73 x 80.
                end_point = 'Conv2d_4a_3x3'
                self.net = slim.conv2d(self.net, depth(192), [3, 3], scope=end_point)
                self.end_points[end_point] = self.net

                # 71 x 71 x 192.
                end_point = 'MaxPool_5a_3x3'
                self.net = slim.max_pool2d(self.net, [3, 3], stride=2, scope=end_point)
                self.end_points[end_point] = self.net

                # 35 x 35 x 192.

            # Inception blocks
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # mixed: 35 x 35 x 256.
                end_point = 'Mixed_5b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_1: 35 x 35 x 288.
                end_point = 'Mixed_5c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv_1_0c_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(64), [1, 1],
                                               scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_2: 35 x 35 x 288.
                end_point = 'Mixed_5d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                               scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                               scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_3: 17 x 17 x 768.
                end_point = 'Mixed_6a'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(384), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                               scope='Conv2d_0b_3x3')
                        branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(self.net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                self.end_points[end_point] = self.net

                # mixed4: 17 x 17 x 768.
                end_point = 'Mixed_6b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_5: 17 x 17 x 768.
                end_point = 'Mixed_6c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_6: 17 x 17 x 768.
                end_point = 'Mixed_6d'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_7: 17 x 17 x 768.
                end_point = 'Mixed_6e'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                               scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                               scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                               scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                               scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_8: 8 x 8 x 1280.
                end_point = 'Mixed_7a'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                               scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                               scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                               padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(self.net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                self.end_points[end_point] = self.net

                # mixed_9: 8 x 8 x 2048.
                end_point = 'Mixed_7b'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat(axis=3, values=[
                          slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                          slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(
                          branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat(axis=3, values=[
                          slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                          slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(
                          branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

                # mixed_10: 8 x 8 x 2048.
                end_point = 'Mixed_7c'
                with tf.variable_scope(end_point):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(self.net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(self.net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat(axis=3, values=[
                          slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                          slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(self.net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(
                          branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat(axis=3, values=[
                          slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                          slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(self.net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(
                          branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                    self.net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
                self.end_points[end_point] = self.net

    def inception_v3(self, inputs,
                     num_classes=5,
                     is_training=True,
                     dropout_keep_prob=0.8,
                     min_depth=16,
                     depth_multiplier=1.0,
                     prediction_fn=slim.softmax,
                     spatial_squeeze=True,
                     reuse=None,
                     create_aux_logits=True,
                     scope='InceptionV3',
                     global_pool=False):

        def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):

            shape = input_tensor.get_shape().as_list()
            if shape[1] is None or shape[2] is None:
                kernel_size_out = kernel_size
            else:
                kernel_size_out = [min(shape[1], kernel_size[0]),
                                   min(shape[2], kernel_size[1])]
            return kernel_size_out

        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):

                # Auxiliary Head logits
                if create_aux_logits and num_classes:
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                        stride=1, padding='SAME'):
                        aux_logits = self.end_points['Mixed_6e']
                        with tf.variable_scope('AuxLogits'):
                            aux_logits = slim.avg_pool2d(
                                aux_logits, [5, 5], stride=3, padding='VALID',
                                scope='AvgPool_1a_5x5')
                            aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                                     scope='Conv2d_1b_1x1')

                            # Shape of feature map before the final layer.
                            kernel_size = _reduced_kernel_size_for_small_input(
                              aux_logits, [5, 5])
                            aux_logits = slim.conv2d(
                              aux_logits, depth(768), kernel_size,
                              weights_initializer=trunc_normal(0.01),
                              padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                            aux_logits = slim.conv2d(
                              aux_logits, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                              scope='Conv2d_2b_1x1')
                            if spatial_squeeze:
                                aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                            self.end_points['AuxLogits'] = aux_logits

                # Final pooling and prediction
                with tf.variable_scope('Logits'):
                    if global_pool:
                        # Global average pooling.
                        self.net = tf.reduce_mean(self.net, [1, 2], keep_dims=True, name='GlobalPool')
                        self.end_points['global_pool'] = self.net
                    else:
                        # Pooling with a fixed kernel size.
                        kernel_size = _reduced_kernel_size_for_small_input(self.net, [8, 8])
                        # TODO(fady) concat textual features
                        self.net = slim.avg_pool2d(self.net, kernel_size, padding='VALID',
                                                   scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                        self.end_points['AvgPool_1a'] = self.net

                    # 1 x 1 x 2048
                    self.net = slim.dropout(self.net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    self.end_points['PreLogits'] = self.net
                    # 2048
                    logits = slim.conv2d(self.net, num_classes, [1, 1], activation_fn=None,
                                         normalizer_fn=None, scope='Conv2d_1c_1x1')
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 1000
                self.end_points['Logits'] = logits
                self.end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    def load_weights(self, checkpoint_path, sess):
        exclusions =['InceptionV3/Logits', 'InceptionV3/AuxLogits']

        flag = 0
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    flag = 1
                    break
            if 'biases' in var.name:
                continue
            elif flag == 1:
                break
            else:
                self.variables_to_restore.append(var)

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, self.variables_to_restore,
                                                 ignore_missing_vars=False)
        init_fn(sess)

