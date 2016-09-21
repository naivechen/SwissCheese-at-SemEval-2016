#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np 

class CNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        # Embedding layer
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name = "W")
        # [None, Sequence_length, embedding_size]
        self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        # [None, sequence_length, embedding_size, channel = 1]
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):

            # Convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
            b_1 = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")
            conv_1 = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W_1,
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "conv_1")
            # [None, sequence_length - filter_size + 1, 1, num_filters]
            h_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b_1), name = "relu")
            # Max-pooling
            pooled_1 = tf.nn.max_pool(
                h_1,
                ksize = [1, 2, 1, 1],
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "pool_1")
            # [None, sequence_length - filter_size, 1, num_filters]
            filter_shape = [2, 1, num_filters, num_filters]
            W_2 = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W1")
            b_2 = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b1")

            conv_2 = tf.nn.conv2d(
                pooled_1,
                W_2,
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "conv_2")
            # [None, sequence_length - filter_size - 1, 1, num_filters]
            h_2 = tf.nn.relu(tf.nn.bias_add(conv_2, b_2), name = "relu_1")
            # [None, 1, 1, 1, num_filters]
            pooled_2 = tf.nn.max_pool(
                h_2,
                ksize = [1, sequence_length - filter_size - 1, 1, 1],
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "pool_2")

            pooled_outputs.append(pooled_2)

        # [[None, 1, 1, num_filters], [None, 1, 1, num_filters], ...]
        num_filters_total = num_filters * len(filter_sizes)
        # [batch, all_pooled_result]
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        W = tf.get_variable(
            "W",
            shape = [num_filters_total, num_classes],
            initializer = tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape = [num_classes]), name = "b")
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
        self.predictions = tf.argmax(self.scores, 1, name = "predictions")

        # CalulateMean Cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        self.loss = tf.reduce_mean(losses)

        # Accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

