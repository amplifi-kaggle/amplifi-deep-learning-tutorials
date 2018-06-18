## -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time
import glob
from PIL import Image

import sys
sys.path.insert(0,'./')
from layers import *

# Network parameters
dropout_keep_prob = 0.5
num_classes = 1000
mean_RGB = [123.68, 116.78, 103.94]

# Network
def VGG16(x):
    x = Conv2D(x, [3, 3, 3, 64], [1, 1, 1, 1], 'SAME', name='conv1_1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu1_1')
    x = Conv2D(x, [3, 3, 64, 64], [1, 1, 1, 1], 'SAME', name='conv1_2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu1_2')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool1')
    pool1 = x
    
    x = Conv2D(x, [3, 3, 64, 128], [1, 1, 1, 1], 'SAME', name='conv2_1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu2_1')
    x = Conv2D(x, [3, 3, 128, 128], [1, 1, 1, 1], 'SAME', name='conv2_2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu2_2')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool2')
    pool2 = x
    
    x = Conv2D(x, [3, 3, 128, 256], [1, 1, 1, 1], 'SAME', name='conv3_1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu3_1')
    x = Conv2D(x, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', name='conv3_2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu3_2')
    x = Conv2D(x, [3, 3, 256, 256], [1, 1, 1, 1], 'SAME', name='conv3_3', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu3_3')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool3')
    pool3 = x
    
    x = Conv2D(x, [3, 3, 256, 512], [1, 1, 1, 1], 'SAME', name='conv4_1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu4_1')
    x = Conv2D(x, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', name='conv4_2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu4_2')
    x = Conv2D(x, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', name='conv4_3', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu4_3')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool4')
    pool4 = x
    
    x = Conv2D(x, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', name='conv5_1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu5_1')
    x = Conv2D(x, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', name='conv5_2', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu5_2')
    x = Conv2D(x, [3, 3, 512, 512], [1, 1, 1, 1], 'SAME', name='conv5_3', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu5_3')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name='pool5')
    pool5 = x
    
    x = Conv2D(x, [7, 7, 512, 4096], [1, 1, 1, 1], 'VALID', name='fc6', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu6')
    fc6 = x
    x = tf.nn.dropout(x, dropout_keep_prob, name='dropout6')
    
    x = Conv2D(x, [1, 1, 4096, 4096], [1, 1, 1, 1], 'VALID', name='fc7', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu7')
    fc7 = x
    x = tf.nn.dropout(x, dropout_keep_prob, name='dropout7')
    
    x = Conv2D(x, [1, 1, 4096, num_classes], [1, 1, 1, 1], 'VALID', name='fc8', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    fc8 = x
    # x = tf.squeeze(x, [1, 2])
    
    return pool1, pool2, pool3, pool4, pool5, fc6, fc7, fc8

# DeepDream
for cnt in range(0, 10):
    # Test image
    img = Image.open('./test'+str(cnt)+'.jpg')
    img = img.convert('RGB')
    img = np.asarray(img)
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] -= mean_RGB[i]
    img_ = img[np.newaxis, :, :, :]
        
    inputs = tf.Variable(img_, name='Input')
    with tf.variable_scope('vgg_16') as scope:    
        pool1, pool2, pool3, pool4, pool5, fc6, fc7, fc8 = VGG16(inputs)

    # Total loss
    loss_total = tf.nn.l2_loss(fc6)

    #
    inputs_var = [v for v in tf.trainable_variables() if 'Input' in v.name]
    vgg16_vars = [v for v in tf.trainable_variables() if v.name.startswith('vgg_16/')]
    
    # Gradient 
    inputs_grad = tf.gradients(loss_total, inputs_var)
        
    # TF saver for VGG16
    saver = tf.train.Saver(vgg16_vars)

    # TF Session
    with tf.Session() as sess:
        # Load or init variables
        tf.global_variables_initializer().run()
        saver.restore(sess, './vgg16_pretrained.ckpt')
        
        # 
        l_total, g_input = sess.run([loss_total, inputs_grad])
        
        g = g_input[0][0,:,:,:]
        img += 2 / np.abs(g).mean() * g
        for c in range(3):
            img[:, :, c] += mean_RGB[c]
        img = np.clip(img, 0, 255).astype(np.uint8)
        res = Image.fromarray(img)
        res.save('./test'+str(cnt+1)+'.jpg')
        
        print('Iter: {:3d} | Loss: {:4.3e}'.format(cnt+1, l_total))
                
    tf.reset_default_graph()
