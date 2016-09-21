#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 
import numpy as np 
import os
import time
import sys

import load_data
import jieba
import datetime

from text_cnn import CNN


# Parameters
# =========================================================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2, 3, 4, 5", "Comma-separated filter sizes (default: '3, 4, 5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every_times", 100, "Evaluate model on cross validation set after this steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_cores", 4, "the cores of the cpu")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
	print "  ", attr.upper(), value
print ""

# =========================================================================================

# Data Preparation
# =========================================================================================
print "Loading data..."
x_train, x_dev, y_train, y_dev, vocab_processor = load_data.get_train_data("chinese")

print "Vocabulary Size: ", len(vocab_processor.vocabulary_)
print "Train/Dev split", len(y_train), len(y_dev)

# =========================================================================================

# Training
# ==================================================================================

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement = FLAGS.allow_soft_placement,
		log_device_placement = FLAGS.log_device_placement,
		inter_op_parallelism_threads = FLAGS.num_cores,
		intra_op_parallelism_threads = FLAGS.num_cores)

	sess = tf.InteractiveSession(config = session_conf)

	with sess.as_default():
		cnn = CNN(
			sequence_length = x_train.shape[1], 
			num_classes = 2,
			vocab_size = len(vocab_processor.vocabulary_), 
			embedding_size = FLAGS.embedding_dim,
			filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters = FLAGS.num_filters)

		train_op = tf.train.AdamOptimizer(1e-4).minimize(cnn.loss)

		# Initialize all variables
		sess.run(tf.initialize_all_variables())

		# Generate batches
		batches = load_data.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

		# Training loop, for each batch...
		i = 1
		for batch in batches:

			x_batch, y_batch = zip(*batch)
			feed_dict = {
				cnn.input_x: x_batch,
				cnn.input_y: y_batch,
				cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
			_, loss, accuracy = sess.run([train_op, cnn.loss, cnn.accuracy],feed_dict)
			
			print "loss ", loss, ", acc ", accuracy
			
			i += 1
			if i % 20 ==0:
				
				feed_dict = {
					cnn.input_x: x_dev,
					cnn.input_y: y_dev,
					cnn.dropout_keep_prob: 1.0}
				loss, accuracy = sess.run([cnn.loss, cnn.accuracy],feed_dict)
			
				print "test: loss ", loss, ", acc ", accuracy
			

