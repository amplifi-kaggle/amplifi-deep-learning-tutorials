# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0,'./')
from layers import *

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


# LeNet-1 Network
def lenet1(x):
    # Reshape from 1D to 2D
    x = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 4 feature maps.
    x = Conv2D(x, [5, 5, 1, 4], [1, 1, 1, 1], 'VALID', 'conv1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu1')
  
    # Pooling layer - downsamples by 2X.
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool1')
  
    # Second convolutional layer -- maps 4 feature maps to 6.
    x = Conv2D(x, [5, 5, 4, 6], [1, 1, 1, 1], 'VALID', 'conv2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu2')
  
    # Second pooling layer.
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool2')
  
    # Fully connected layer 1:
    # After 2 round of downsampling, our 28x28x1 image is down to 4x4x6 feature maps
    # Maps this to 10 features.
    x = tf.reshape(x, [-1, 4*4*6])
    x = Dense(x, [4*4*6, 10], name='fc3', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    
    return x


# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Input image
x = tf.placeholder(tf.float32, [None, 784])
# Real label
y_ = tf.placeholder(tf.float32, [None, 10])

# Build the graph for the LeNet-1, output of the network
y = lenet1(x)

# Cross entropy 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Count correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
# GD
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)
## Momentum
#train_step = tf.train.MomentumOptimizer(1e-3).minimize(cross_entropy)
## RMSProp
#train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)
## Adam
# train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)


with tf.Session() as sess:
    # Variable initialization
    sess.run(tf.global_variables_initializer())
    
    # Run 1200 iterations
    for i in range(1200):
        # Read 50 data from mnist (=the size of mini-batch is 50)
        batch = mnist.train.next_batch(50)
        
        # Print out training acc. every 100 iterations
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            
        # Train
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
