# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

## Initializations
xavier_normal_init = tf.contrib.layers.xavier_initializer(uniform=False)
he_normal_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)

## Layers
def Dense(input, weight_shape, name='Dense', W_init=he_normal_init, mean=0.0, stddev=0.01, b_init=0.0):
    with tf.variable_scope(name):
        if W_init == 'Gaussian':
            W = tf.Variable(tf.random_normal(weight_shape, mean=mean, stddev=stddev), name="W")
        else:
            W = tf.get_variable("W", weight_shape, initializer=he_normal_init)
        b = tf.get_variable("b", (weight_shape[-1]),initializer=tf.constant_initializer(value=b_init))
        
    return tf.matmul(input, W) + b
    
def Conv2D(input, kernel_shape, strides, padding, name='Conv2d', W_init=he_normal_init, mean=0.0, stddev=0.01, b_init=0.0):
    with tf.variable_scope(name):
        if W_init == 'Gaussian':
            W = tf.Variable(tf.random_normal(kernel_shape, mean=mean, stddev=stddev), name="W")
        else:
            W = tf.get_variable("W", kernel_shape, initializer=W_init)
        b = tf.get_variable("b", (kernel_shape[-1]),initializer=tf.constant_initializer(value=b_init))
    
    return tf.nn.conv2d(input, W, strides, padding) + b

## Regularizations
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

## Activations
def LeakyReLU(input, alpha=0.2):
    return tf.maximum(input, alpha*input)


