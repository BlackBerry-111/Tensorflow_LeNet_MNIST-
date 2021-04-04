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
from tf_fun import *
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x  = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])

# input data shape [batch, in_height, in_width, in_channels]
x_imag = tf.reshape(x, [-1, 28, 28, 1])

"""1st Convolution Layer"""
# input filter shape [filter_height, filter_width, in_channels, out_channels]
w_conv1 = weigh_variable([5,5,1,6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_imag, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


"""2nd Convolution Layer"""
# input filter shape [filter_height, filter_width, in_channels, out_channels]
w_conv2 = weigh_variable([5,5,6,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


"""1st Fully Connected Layer"""
w_fc1 = weigh_variable([7*7*16,120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

"""2st Fully Connected Layer"""
w_fc2 = weigh_variable([120,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2)+b_fc2)

"""Loss Calculation and Parameter Optimization"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step    = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()

"""Test Accuracy"""
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

"""Initial all parameter"""
sess.run(tf.global_variables_initializer())

"""MNIST Data"""
mnist_data = mnist_img()
mnist_data.update_train_img()
c = []



"""Training"""
start_time = time.time()
for i in range(20000):
    # Training Data Acquirezition
    batch_xs, batch_ys = mnist_data.update_train_batch(batch=50)
    # Accuracy test every 100 batch
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        c.append(train_accuracy)
        train_loss     = cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        print("training loss %g" %train_loss)
        end_time = time.time()
        print('time: ',(end_time-start_time))
        strat_time = time.time()
        
    train_step.run(feed_dict={x:batch_xs, y_:batch_ys})

sess.close()

plt.plot(c)
plt.tight_layout()
plt.savefig('tf_lenet_relu_mnist.jpg', dpi=200)











