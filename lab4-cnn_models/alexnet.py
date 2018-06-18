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
B = 4 # batch
H = 227 # height of image
W = 227 # width
C = 3 # channel (R,G,B)
dropout_keep_prob = 0.5
num_classes = 1000
momentum = 0.9
weight_decay = 0.0005
lr_init = 0.01
lr_decay = 0.1
mean_RGB = [123, 117, 104]  # For data normalization
phase = 'Train'


# AlexNet Network
def AlexNet(x):
    x = Conv2D(x, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', name='conv1', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu1')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', name='pool1')
    x = tf.nn.lrn(x, bias=2, alpha=1e-4, beta=0.75, name='lrn1')
    
    x = Conv2D(tf.pad(x, [[0,0], [2,2], [2,2], [0,0]], 'CONSTANT'), [5, 5, 96, 256], [1, 1, 1, 1], 'VALID', name='conv2', W_init='Gaussian', mean=0, stddev=0.01, b_init=1)
    x = tf.nn.relu(x, name='relu2')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', name='pool2')
    x = tf.nn.lrn(x, bias=2, alpha=1e-4, beta=0.75, name='lrn2')

    x = Conv2D(tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'CONSTANT'), [3, 3, 256, 384], [1, 1, 1, 1], 'VALID', name='conv3', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu3')
    
    x = Conv2D(tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'CONSTANT'), [3, 3, 384, 384], [1, 1, 1, 1], 'VALID', name='conv4', W_init='Gaussian', mean=0, stddev=0.01, b_init=1)
    x = tf.nn.relu(x, name='relu4')
    
    x = Conv2D(tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'CONSTANT'), [3, 3, 384, 256], [1, 1, 1, 1], 'VALID', name='conv5', W_init='Gaussian', mean=0, stddev=0.01, b_init=1)
    x = tf.nn.relu(x, name='relu5')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID', name='pool5')
    
    s_x = x.get_shape().as_list()
    x = tf.reshape(x, [-1, s_x[1]*s_x[2]*s_x[3]])
    s_x = x.get_shape().as_list()
    x = Dense(x, [s_x[1], 4096], name='fc6', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu6')
    x = tf.nn.dropout(x, dropout_keep_prob, name='dropout6')
    
    x = Dense(x, [4096, 4096], name='fc7', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    x = tf.nn.relu(x, name='relu7')
    x = tf.nn.dropout(x, dropout_keep_prob, name='dropout7')
    
    x = Dense(x, [4096, num_classes], name='fc8', W_init='Gaussian', mean=0, stddev=0.01, b_init=0)
    
    return x

# Whole model, create a graph
inputs = tf.placeholder(tf.float32, shape=[None, H, W, C]) # shape=[B, H, W, C]
labels = tf.placeholder(tf.int32, shape=[None])
lr = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope('AlexNet') as scope:
    outputs = AlexNet(inputs)

# Weight decay
trainable_Ws = [v for v in tf.trainable_variables() if v.name.endswith('/W')]
loss_l2_reg = 0
for w in trainable_Ws:
    loss_l2_reg += tf.nn.l2_loss(w)
    
# Cross entropy
loss_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs))

# Total loss
loss_total = tf.reduce_mean(loss_ce + weight_decay*loss_l2_reg)

# Momentum optimizer
if phase == 'Train':
    trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith('AlexNet/')]
    train_alexnet = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_total, var_list=trainable_vars)

# Compute top 5 class and accuracy
result_top5 = tf.nn.top_k(tf.nn.softmax(outputs), 5)

# Data preparation
# Training images
train_filenames = glob.glob('./imagenet_train/*.JPEG')
train_images = np.zeros((len(train_filenames), H, W, C), dtype=np.uint8)
for i in range(len(train_filenames)):
    img = Image.open(train_filenames[i])
    img = img.convert('RGB')
    img = np.asarray(img)
    # Random crop
    r_y = np.random.randint(img.shape[0] - H)
    r_x = np.random.randint(img.shape[1] - W)
    img = img[r_y:r_y+H, r_x:r_x+W, :]
    # Random LR flip
    if np.random.random() < 0.5:
        img = np.copy(img[:, ::-1, :])
    img.setflags(write=True)
    # Random color transform
    r_rgb = 0.1 * np.random.randn(3) * 255
    r_rgb = r_rgb.astype(np.uint8)
    for c in range(3):
        img[:, :, c] += r_rgb[c]
    train_images[i, :, :, :] = img
train_images = train_images.astype(np.float32)
for i in range(3):
    train_images[:, :, :, i] -= mean_RGB[i]
num_training_samples = train_images.shape[0]
# Training labels
train_labels = np.loadtxt('./imagenet_train/labels.txt').astype(np.int)

# Validation images
val_filenames = glob.glob('./imagenet_val/*.JPEG')
val_images = np.zeros((len(val_filenames), H, W, C), dtype=np.uint8)
for i in range(len(val_filenames)):
    img = Image.open(val_filenames[i])
    img = img.convert('RGB')
    img = np.asarray(img)
    # Center crop
    s_y = (img.shape[0] - H) // 2
    s_x = (img.shape[1] - W) // 2
    img = img[s_y:s_y+H, s_x:s_x+W, :]
    val_images[i, :, :, :] = img
val_images = val_images.astype(np.float32)
for i in range(3):
    val_images[:, :, :, i] -= mean_RGB[i]
num_val_samples = val_images.shape[0]
# Validation labels
val_labels = np.loadtxt('./imagenet_val/labels.txt').astype(np.int)

# Test images
test_filenames = glob.glob('./imagenet_test/*.JPEG')
test_images = np.zeros((len(test_filenames), H, W, C), dtype=np.uint8)
for i in range(len(test_filenames)):
    img = Image.open(test_filenames[i])
    img = img.convert('RGB')
    img = np.asarray(img)
    # Center crop
    s_y = (img.shape[0] - H) // 2
    s_x = (img.shape[1] - W) // 2
    img = img[s_y:s_y+H, s_x:s_x+W, :]
    test_images[i, :, :, :] = img
test_images = test_images.astype(np.float32)
for i in range(3):
    test_images[:, :, :, i] -= mean_RGB[i]
num_test_samples = test_images.shape[0]
# Test labels
test_labels = np.loadtxt('./imagenet_test/labels.txt').astype(np.int)

# TF saver
saver = tf.train.Saver()

# TF Session
with tf.Session() as sess:
    # Load or init variables
    if phase == 'Train':
        tf.global_variables_initializer().run()
    else:
        saver.restore(sess, './alexnet_models/alexnet_e1.ckpt')
        print("Model restored.")
    
    if phase == 'Train':
        e = 0 # epoch
        p = 0 # pointer
        lr_curr = lr_init  # learning rate
        
        # Training
        for i in range(0, 20):
            t = time.time()
            l_total, _ = sess.run([loss_total, train_alexnet], feed_dict={inputs: train_images[p:p+B], labels: train_labels[p:p+B], lr:lr_curr})
            dT = time.time() - t
            print('Train Epoch: {:3d} | Iter: {:4d} | Loss: {:4.3e} | dT: {:4.3f}s'.format(e, i, l_total, dT))
            
            p += B
            if p >= num_training_samples:
                l_val = 0
                for j in range(0, num_val_samples):
                    l_total = sess.run(loss_total, feed_dict={inputs: val_images[j:j+1], labels: val_labels[j:j+1], lr:lr_curr})
                    l_val += l_total
                print('Val   Epoch: {:3d} | Loss: {:4.3e}'.format(e, l_val))
                
                e += 1
                p = 0
            
                save_path = saver.save(sess, './alexnet_models/alexnet_e'+str(e)+'.ckpt')
                print("Model saved in file: %s" % save_path)
            
            # Adjust learning rate (optional)
            if e >= 20:
                lr_curr *= lr_decay

    # Test
    for i in range(0, num_test_samples):
        t = time.time()
        top5 = sess.run(result_top5, feed_dict={inputs: test_images[i:i+1]})
        dT = time.time() - t
        top5_acc = top5[0][0]
        top5_category = top5[1][0]
        print('Test  Image #: {:3d} | Answer :{:3d} | Top5 category/acc: {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} | dT: {:4.3f}s'.format(i, test_labels[i], top5_category[0],top5_acc[0], top5_category[1],top5_acc[1], top5_category[2],top5_acc[2], top5_category[3],top5_acc[3], top5_category[4],top5_acc[4], dT))
