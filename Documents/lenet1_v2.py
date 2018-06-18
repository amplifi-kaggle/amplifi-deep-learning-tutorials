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
    # First convolutional layer - maps one grayscale image to 4 feature maps.
    x = Conv2D(x, [5, 5, 1, 4], [1, 1, 1, 1], 'VALID', 'conv1')
    x = tf.nn.relu(x, name='relu1')
    x = BatchNorm(x, is_train, name='bn1')
  
    # Pooling layer - downsamples by 2X.
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool1')
  
    # Second convolutional layer -- maps 4 feature maps to 6.
    x = Conv2D(x, [5, 5, 4, 6], [1, 1, 1, 1], 'VALID', 'conv2')
    x = tf.nn.relu(x, name='relu2')
    x = BatchNorm(x, is_train, name='bn2')
  
    # Second pooling layer.
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool2')
  
    # Fully connected layer 1:
    # After 2 round of downsampling, our 28x28x1 image is down to 4x4x6 feature maps
    # Maps this to 10 features.
    x = tf.reshape(x, [-1, 4*4*6])
    x = Dense(x, [4*4*6, 10], name='fc3')
    
    return x


# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Input image
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
# Real label
y_ = tf.placeholder(tf.float32, [None, 10])
is_train = tf.placeholder(tf.bool, [])

# Build the graph for the LeNet-1, output of the network
y = lenet1(x)

# Weight decay
trainable_Ws = [v for v in tf.trainable_variables() if v.name.endswith('/W')]
loss_l2_reg = 0
for w in trainable_Ws:
    loss_l2_reg += tf.nn.l2_loss(w)

# Cross entropy 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Total loss
loss_total = tf.reduce_mean(cross_entropy + 0.0001*loss_l2_reg)

# Count correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimizer
## GD
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
## Momentum
#train_step = tf.train.MomentumOptimizer(1e-3, 0.9).minimize(cross_entropy)
## RMSProp
#train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)
# Adam
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)


with tf.Session() as sess:
    # Variable initialization
    sess.run(tf.global_variables_initializer())
    
    # Run 1200 iterations
    for i in range(1200):
        # Read 50 data from mnist (=the size of mini-batch is 50)
        batch_x, batch_y = mnist.train.next_batch(50)    
        
        # Reshape: (50,28*28*1) -> (50,28,28,1)
        batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
        
        # Random LR flip
        if np.random.random() < 0.5:
            batch_x = batch_x[:, :, ::-1, :]
            
        # Random rotation
        batch_x = np.transpose(batch_x, [1, 2, 0, 3])
        batch_x = np.rot90(batch_x, np.random.randint(4))
        batch_x = np.transpose(batch_x, [2, 0, 1, 3])
            
        # Print out training acc. every 100 iterations
        if i % 100 == 0:
            train_accuracy, train_loss = sess.run([accuracy, cross_entropy], feed_dict={x: batch_x, y_: batch_y, is_train: False})
            print('step %d, training accuracy %g, cross_entropy %g' % (i, train_accuracy, train_loss))
            
        # Train
        train_step.run(feed_dict={x: batch_x, y_: batch_y, is_train: True})
  
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_train: False}))
