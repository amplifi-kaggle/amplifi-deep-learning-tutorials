# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

'''
Nowadays

Data augmentatino
He initialization with ReLU
Fully convolutinal architecture
     Avoid fully connected layers
Batch norm
     No dropout
     No weight decay
Adam optimizer
'''

# Xavier Initializer
xavier_normal_init = tf.contrib.layers.xavier_initializer(uniform=False)

# He Initializer (Kaiming He)
he_normal_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

# But random gaussian initialization is not good at deep networks
# Layers
def Dense(input, weight_shape, name='Dense', W_init=he_normal_init, mean=0.0, stddev=0.01, b_init=0.0):
    with tf.variable_scope(name):
        if W_init == 'Gaussian':
            W = tf.Variable(tf.random_normal(weight_shape, mean=mean, stddev=stddev), name="W")
        else:
            W = tf.get_variable("W", weight_shape, initializer=he_normal_init)
        b = tf.get_variable("b", weight_shape[-1], initializer=tf.constant_initializer(value=b_init))
        
    return tf.matmul(input, W) + b
    
# Convolutional layer
def Conv2D(input, kernel_shape, strides, padding, name='Conv2d', W_init=he_normal_init, mean=0.0, stddev=0.01, b_init=0.0):
    with tf.variable_scope(name):
        if W_init == 'Gaussian':
            W = tf.Variable(tf.random_normal(kernel_shape, mean=mean, stddev=stddev), name="W")
        else:
            W = tf.get_variable("W", kernel_shape, initializer=W_init)
        b = tf.get_variable("b", kernel_shape[-1], initializer=tf.constant_initializer(value=b_init))
    return tf.nn.conv2d(input, W, strides, padding) + b
    

# Regularizations
def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]
    
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        
        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3)

# Activations
def LeakyReLU(input, alpha=0.2):
    return tf.maximum(input, alpha*input)

# Define network

def Network(t):
    t = Conv2D(t, kernel_shape=[3, 3, 1, 16], strides=[1, 1, 1, 1], padding='SAME', name='Conv1')
    t = tf.nn.relu(t)
    
    for i in range(1):
        t = Conv2D(t, kernel_shape=[3, 3, 16, 16], strides=[1, 1, 1, 1], padding='SAME', name='Conv' + str(i+2))
        t = tf.nn.relu(t)
        
    t = Conv2D(t, kernel_shape=[3, 3, 16, 4], strides=[1, 1, 1, 1], padding='SAME', name='Conv12')
    t = tf.nn.relu(t)
    
    t = tf.reshape(t, [-1, 28 * 28 * 4])
    t = Dense(t, [28*28*4, 10], name='Dense1')
    
    return t


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


from PIL import Image
import numpy as np

# read image
def np_from_img(fname):
    return np.asarray(Image.open(fname), dtype=np.float32)

# write image
def save_as_img(ar, fname):
    Image.fromarray(ar.round().astype(np.uint8)).save(fname)
    
# normalization
def norm(ar):
    return 255*np.absolute(ar)/np.max(ar)



# placeholder
x  = tf.placeholder("float", [None, 28, 28, 1])
y_ = tf.placeholder("float", [None, 10])
learning_rate = tf.placeholder("float", shape=[])

# get result of network
y = Network(x)

# optimization
# softmax + cross_entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# make session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# set init learning rate, batch_size
batch_size = 100
lr = 0.01

# load valid data
test_xs, test_ys = mnist.test.images, mnist.test.labels
print(test_xs.shape)
test_xs = np.reshape(test_xs, [-1, 28, 28, 1]) # Reshape data shape
print(test_xs.shape)

# save 'img_len' valid images
str_idx = 0
img_len = 20
# get first 20 images from test_xs
input_image = test_xs[str_idx: str_idx + img_len, :, :, 0]

canvas = np.zeros([28, 28 * img_len])

for i in range(img_len):
    canvas[:, 28 * i : 28 * (i + 1)] = input_image[i, :, :]

save_as_img(norm(canvas), './canvas.png')

for step in range(1000):
    
    # step learning policy
    if(step % 100 == 0 and step != 0):
        lr = lr / 2
        
    # load datasets
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1]) # Reshape data shape
    
    # run session
    sess.run(train_step, feed_dict = {x: batch_xs, y_ : batch_ys, learning_rate: lr})
    
    # validation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_ = sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})
    
    print('step: {:01d} | accuracy: {:.4f} | learning rate : {:.4f}'.format(step, float(accuracy_), lr))
        
    if (step % 10 == 0):
        mnist_img = np.reshape(input_image, [-1, 28, 28, 1])
        mnist_label = sess.run(y, feed_dict={x: mnist_img})
        print("label results: ", np.argmax(mnist_label, 1))
    
    
    
    






