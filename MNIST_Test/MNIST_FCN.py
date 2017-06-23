# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def fcn(x):
  # Reshape to use within a convolutional neural net.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Third  convolutional layer-- maps 64 feature maps to 120.
  W_conv3 = weight_variable([7 , 7 , 64, 120])
  b_conv3 = bias_variable([120])
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides=[1,1,1,1],padding='VALID') + b_conv3)
  
  # Fourth  convolutional layer-- maps 120 feature maps to 84.
  W_conv4 = weight_variable([1 , 1 , 120, 84])
  b_conv4 = bias_variable([84])
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4,strides=[1,1,1,1],padding='VALID') + b_conv4)

  # Fifth  convolutional layer-- maps 84 feature maps to 10.
  W_conv5 = weight_variable([1 , 1 , 84, 10])
  b_conv5 = bias_variable([10])
  h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5,strides=[1,1,1,1],padding='VALID') + b_conv5)
  
  # Reshape to match with y_
  y_conv =  tf.reshape(h_conv5, [-1, 10])
  
  return y_conv


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


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
  y_conv = fcn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
      batch = mnist.train.next_batch(128)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print( "Optimization Finished!")
    test_len =  1000
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images[:test_len], y_: mnist.test.labels[:test_len]}))

if __name__ == '__main__':
  tf.app.run()