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
H = 224 # height of image
W = 224 # width
C = 3 # channel (R,G,B)
num_classes = 1000
momentum = 0.9
weight_decay = 0.0002
lr_init = 0.01
lr_decay = 0.96
mean_RGB = [123.68, 116.78, 103.94]
phase = 'Train'

# Inception module
def Inception(x, c11, p33, c33, p55, c55, pPool):
    s_x = x.get_shape().as_list()
    
    # 1x1 conv
    x11 = Conv2D(x, [1, 1, s_x[3], c11], [1, 1, 1, 1], 'SAME', name='conv11')
    x11 = tf.nn.relu(x11, name='relu11')

    # 3x3 conv
    x33 = Conv2D(x, [1, 1, s_x[3], p33], [1, 1, 1, 1], 'SAME', name='conv33_p')
    x33 = tf.nn.relu(x33, name='relu33_p')
    x33 = Conv2D(x33, [3, 3, p33, c33], [1, 1, 1, 1], 'SAME', name='conv33')
    x33 = tf.nn.relu(x33, name='relu33')
    
    # 5x5 conv
    x55 = Conv2D(x, [1, 1, s_x[3], p55], [1, 1, 1, 1], 'SAME', name='conv55_p')
    x55 = tf.nn.relu(x55, name='relu55_p')
    x55 = Conv2D(x55, [5, 5, p55, c55], [1, 1, 1, 1], 'SAME', name='conv55')
    x55 = tf.nn.relu(x55, name='relu55')
    
    # Max pooling
    xP = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME', name='pool')
    xP = Conv2D(xP, [1, 1, s_x[3], pPool], [1, 1, 1, 1], 'SAME', name='pool_p')
    xP = tf.nn.relu(xP, name='reluP')

    # Concatenation
    x = tf.concat([x11, x33, x55, xP], axis=3)

    return x
    
# Auxiliary classifier
def AuxOut(x):
    x = tf.nn.avg_pool(x, [1, 5, 5, 1], [1, 3, 3, 1], 'VALID', name='pool1')
    
    s_x = x.get_shape().as_list()
    x = Conv2D(x, [1, 1, s_x[3], 128], [1, 1, 1, 1], 'SAME', name='conv2')
    x = tf.nn.relu(x, name='relu2')
    
    x = Conv2D(x, [4, 4, 128, 1024], [1, 1, 1, 1], 'VALID', name='fc3')
    x = tf.nn.relu(x, name='relu3')
    x = tf.nn.dropout(x, 0.3, name='dropout3')
    
    x = Conv2D(x, [1, 1, 1024, 1000], [1, 1, 1, 1], 'VALID', name='fc4')
    x = tf.squeeze(x, [1, 2])
    
    return x    
  
# Network  
def GoogLeNet(x):
    x = Conv2D(x, [7, 7, 3, 64], [1, 2, 2, 1], 'SAME', name='conv1')
    x = tf.nn.relu(x, name='relu1')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool1')
    x = tf.nn.lrn(x, name='lrn1')
    
    x = Conv2D(x, [1, 1, 64, 64], [1, 1, 1, 1], 'SAME', name='conv2a')
    x = tf.nn.relu(x, name='relu2a')
    x = Conv2D(x, [3, 3, 64, 192], [1, 1, 1, 1], 'SAME', name='conv2b')
    x = tf.nn.relu(x, name='relu2b')
    x = tf.nn.lrn(x, name='lrn2')
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool2')
    
    with tf.variable_scope('Inception3a'):
        x = Inception(x, 64, 96, 128, 16, 32, 32)
    with tf.variable_scope('Inception3b'):
        x = Inception(x, 128, 128, 192, 32, 96, 64)
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool3')
    
    with tf.variable_scope('Inception4a'):
        x = Inception(x, 192, 96, 208, 16, 48, 64)
        x0 = AuxOut(x)
    with tf.variable_scope('Inception4b'):
        x = Inception(x, 160, 112, 224, 24, 64, 64)
    with tf.variable_scope('Inception4c'):
        x = Inception(x, 128, 128, 256, 24, 64, 64)
    with tf.variable_scope('Inception4d'):
        x = Inception(x, 112, 144, 288, 32, 64, 64)
        x1 = AuxOut(x)
    with tf.variable_scope('Inception4e'):
        x = Inception(x, 256, 160, 320, 32, 128, 128)
    x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool4')
    
    with tf.variable_scope('Inception5a'):
        x = Inception(x, 256, 160, 320, 32, 128, 128)
    with tf.variable_scope('Inception5b'):
        x = Inception(x, 384, 192, 384, 48, 128, 128)
        
    x = tf.nn.avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], 'VALID', name='pool5')
    x = tf.nn.dropout(x, 0.6, name='dropout5')
    x = Conv2D(x, [1, 1, 1024, num_classes], [1, 1, 1, 1], 'VALID', name='fc5')
    x = tf.squeeze(x, [1, 2])
    x2 = x
        
    return x0, x1, x2

# Whole model
inputs = tf.placeholder(tf.float32, shape=[None, H, W, C])
labels = tf.placeholder(tf.int32, shape=[None])
lr = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope('GoogLeNet') as scope:
    output0, output1, output2 = GoogLeNet(inputs)

# Weight decay
trainable_Ws = [v for v in tf.trainable_variables() if v.name.endswith('/W')]
loss_l2_reg = 0
for w in trainable_Ws:
    loss_l2_reg += tf.nn.l2_loss(w)
    
# Cross entropy
loss_ce0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output0)
loss_ce1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output1)
loss_ce2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output2)
loss_ce = tf.reduce_mean([loss_ce0, loss_ce1, loss_ce2])

# Total loss
loss_total = tf.reduce_mean(loss_ce + weight_decay*loss_l2_reg)

# Momentum optimizer
if phase == 'Train':
    trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith('GoogLeNet/')]
    train_googlenet = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_total, var_list=trainable_vars)

# Compute top 5 class and accuracy
result_top5 = tf.nn.top_k(tf.nn.softmax(output2), 5)

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
        saver.restore(sess, './googlenet_models/googlenet_e1.ckpt')
        print("Model restored.")
    
    if phase == 'Train':
        e = 0 # epoch
        p = 0 # pointer
        lr_curr = lr_init  # learning rate
        
        # Training
        for i in range(0, 20):
            t = time.time()
            l_total,  _= sess.run([loss_total, train_googlenet], feed_dict={inputs: train_images[p:p+B], labels: train_labels[p:p+B], lr:lr_curr})
            dT = time.time() - t
            print('Epoch: {:3d} | Iter: {:4d} | Loss: {:4.3e} | dT: {:4.3f}s'.format(e, i, l_total, dT))
            
            p += B
            if p >= num_training_samples:
                l_val = 0
                for j in range(0, num_val_samples):
                    l_total = sess.run(loss_total, feed_dict={inputs: val_images[j:j+1], labels: val_labels[j:j+1], lr:lr_curr})
                    l_val += l_total
                print('Val   Epoch: {:3d} | Loss: {:4.3e}'.format(e, l_val))
                
                e += 1
                p = 0
                
                save_path = saver.save(sess, './googlenet_models/googlenet_e'+str(e)+'.ckpt')
                print("Model saved in file: %s" % save_path)
            
                # Adjust learning rate (optional)
                if e % 8 == 0:
                    lr_curr *= lr_step
        
    # Test
    for i in range(0, num_test_samples):
        t = time.time()
        top5 = sess.run(result_top5, feed_dict={inputs: test_images[i:i+1]})
        dT = time.time() - t
        top5_acc = top5[0][0]
        top5_category = top5[1][0]
        print('Test image #: {:3d} | Answer :{:3d} | Top5 category/acc: {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} {:3d}/{:.3f} | dT: {:4.3f}s'.format(i, test_labels[i], top5_category[0],top5_acc[0], top5_category[1],top5_acc[1], top5_category[2],top5_acc[2], top5_category[3],top5_acc[3], top5_category[4],top5_acc[4], dT))

