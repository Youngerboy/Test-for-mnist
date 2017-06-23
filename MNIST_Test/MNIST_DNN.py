# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def dnn(x):

  W_hidden = weight_variable([784, 500])
  b_hidden = bias_variable([500])

  h_hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

  keep_prob = tf.placeholder(tf.float32)
  h_hidden_drop = tf.nn.dropout(h_hidden, keep_prob)
  
  W_output= weight_variable([500, 10])
  b_output= bias_variable([10])

  y = tf.matmul(h_hidden_drop, W_output) + b_output
  return y, keep_prob

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y , keep_prob= dnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
      batch = mnist.train.next_batch(128)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print( "Optimization Finished!")
    test_len =  1000
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images[:test_len], y_: mnist.test.labels[:test_len], keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run()