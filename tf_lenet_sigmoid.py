# LeNet test of MNIST by TensorFlow
# number of all train image: 60,000
# number of all test image : 10,000
# Software: Python3
# Designer: Black Chocolate


# from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from mnist import mnist_img
# import numpy as np
import time

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# sess = tf.Session()

# test_label, test_img = data_mnist('MNIST', 'test')
# train_label, train_img = data_mnist('MNIST', 'train')


# hello = tf.constant('hello world')
# print(sess.run(hello))

x  = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])

# input data shape [batch, in_height, in_width, in_channels]
x_imag = tf.reshape(x, [-1, 28, 28, 1])

"""1st Convolution Layer"""
# input filter shape [filter_height, filter_width, in_channels, out_channels]
filter1 = tf.Variable(tf.truncated_normal([5,5,1,6]))
bias1   = tf.Variable(tf.truncated_normal([6]))
conv1   = tf.nn.conv2d(x_imag, filter1, strides=[1,1,1,1], padding='SAME')
# h_conv1 = tf.nn.relu(conv1+bias1)
h_conv1 = tf.nn.sigmoid(conv1+bias1)


"""Pooling layer"""
# ksize  [1, pool_height, pool_width, 1]
# stride [1, v_stride, h_stride, 1]
maxPool2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

"""2nd Convolution Layer"""
# input filter shape [filter_height, filter_width, in_channels, out_channels]
filter2 = tf.Variable(tf.truncated_normal([5,5,6,16]))
bias2   = tf.Variable(tf.truncated_normal([16]))
conv2   = tf.nn.conv2d(maxPool2, filter2, strides=[1,1,1,1], padding='SAME')
# h_conv2 = tf.nn.relu(conv2+bias2)
h_conv2 = tf.nn.sigmoid(conv2+bias2)

"""Pooling layer"""
# ksize  [batch, in_height, in_width, in_channels]
# stride [1, stride, stride, 1]
maxPool3 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

"""3rd Convolution Layer"""
# input filter shape [filter_height, filter_width, in_channels, out_channels]
filter3 = tf.Variable(tf.truncated_normal([5,5,16,120]))
bias3   = tf.Variable(tf.truncated_normal([120]))
conv3   = tf.nn.conv2d(maxPool3, filter3, strides=[1,1,1,1], padding='SAME')
# h_conv3 = tf.nn.relu(conv3+bias3)
h_conv3 = tf.nn.sigmoid(conv3+bias3)

"""1st Fully Connected Layer"""
W_fc1 = tf.Variable(tf.truncated_normal([7*7*120, 80]))
b_fc1 = tf.Variable(tf.truncated_normal([80]))
h_pool2_flat = tf.reshape(h_conv3, [-1,7*7*120])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

"""2st Fully Connected Layer"""
W_fc2 = tf.Variable(tf.truncated_normal([80, 10]))
b_fc2 = tf.Variable(tf.truncated_normal([10]))
# y_conv = tf.maximum(tf.nn.softmax(tf.matmul(h_fc1, W_fc2)+b_fc2), 1e-30)
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2)+b_fc2)

"""Loss Calculation and Parameter Optimization"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step    = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession()
# sess = tf.Session()

"""Test Accuracy"""
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

"""Initial all parameter"""
sess.run(tf.global_variables_initializer())

"""MNIST Data"""
mnist_data = mnist_img()
mnist_data.update_train_img()
mnist_data.update_test_img()

# mnist_data_set = input_data.read_data_sets('MNIST_data', one_hot=True)

"""Training"""
start_time = time.time()
for i in range(10000):
    #Training Data Acquirezition
    batch_xs, batch_ys = mnist_data.update_train_batch()
    # batch_xs, batch_ys = mnist_data_set.train.next_batch(200)
    #Accuracy test every 100 batch
    if i%10 == 0:
        # test_xs, test_ys = mnist_data.update_test_batch()
        # train_accuracy = accuracy.eval(feed_dict={x: test_xs, y_: test_ys})
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        train_loss     = cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        print("training loss %g" %train_loss)
        
        end_time = time.time()
        print('time: ',(end_time-start_time))
        strat_time = time.time()
        # result = y_conv.eval(feed_dict={x: batch_xs, y_: batch_ys})
        # print(result)
        
    train_step.run(feed_dict={x:batch_xs, y_:batch_ys})


sess.close()













