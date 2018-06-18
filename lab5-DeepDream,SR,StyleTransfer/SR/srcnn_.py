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

# Compute PSNR
def PSNR(y_true, y_pred, shave_border=4, maxVal=255):
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)
    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(maxVal/rmse)

# RGB2YCbCr
def _rgb2ycbcr(img, maxVal=255):
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])
    if maxVal == 1:
        O = O / 255.0
    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    return ycbcr

# Network parameters
B = 4
H = 32
W = 32
C = 1
r = 2    # scale factor for SR
lr_init = 0.0001
momentum = 0.9
phase = 'Train'

# Network
def SRCNN(x):
    x = Conv2D(x, [9, 9, 1, 64], [1, 1, 1, 1], 'VALID', name='conv1', W_init='Gaussian', mean=0, stddev=0.001, b_init=0)
    x = tf.nn.relu(x, name='relu1')

    x = Conv2D(x, [1, 1, 64, 32], [1, 1, 1, 1], 'VALID', name='conv2', W_init='Gaussian', mean=0, stddev=0.001, b_init=0)
    x = tf.nn.relu(x, name='relu2')

    x = Conv2D(x, [5, 5, 32, 1], [1, 1, 1, 1], 'VALID', name='conv3', W_init='Gaussian', mean=0, stddev=0.001, b_init=0)

    return x

# Whole model
inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1])
labels = tf.placeholder(tf.float32, shape=[None, None, None, 1])
lr = tf.placeholder(tf.float32, shape=[])

with tf.variable_scope('SRCNN') as scope:
    outputs = SRCNN(inputs)
    
# Mean squared error
loss = tf.reduce_mean(tf.square(outputs - labels[:, 6:-6, 6:-6, :]))

# Momentum optimizer
if phase == 'Train':
    trainable_vars_last_layer = [v for v in tf.trainable_variables() if 'conv3' in v.name]
    trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith('SRCNN/') and not v in trainable_vars_last_layer]

    opt1 = tf.train.MomentumOptimizer(lr, momentum)
    opt2 = tf.train.MomentumOptimizer(lr*0.1, momentum)
    #opt1 = tf.train.AdamOptimizer()
    #opt2 = tf.train.AdamOptimizer()

    grads = tf.gradients(loss, trainable_vars + trainable_vars_last_layer)
    grads1 = grads[:len(trainable_vars)]
    grads2 = grads[len(trainable_vars):]

    train_op1 = opt1.apply_gradients(zip(grads1, trainable_vars))
    train_op2 = opt2.apply_gradients(zip(grads2, trainable_vars_last_layer))
    train_SRCNN = tf.group(train_op1, train_op2)

# Data preparation
def PrepareTrainImages():
    # Training images
    train_filenames = glob.glob('./train_91/*.bmp')
    train_labels = np.zeros((len(train_filenames), H, W, C), dtype=np.uint8)
    train_images = np.zeros((len(train_filenames), H, W, C), dtype=np.uint8)
    for i in range(len(train_filenames)):
        img = Image.open(train_filenames[i])
        img = img.convert('RGB')
        # Bicubic down-upsampling
        w, h = img.size
        img_input = img.resize((w//r, h//r), Image.ANTIALIAS)
        img_input = img_input.resize((w, h), Image.BICUBIC)
        img = np.asarray(img)
        img_input = np.asarray(img_input)
        # Random crop
        r_y = np.random.randint(img.shape[0] - H)
        r_x = np.random.randint(img.shape[1] - W)
        img = img[r_y:r_y+H, r_x:r_x+W, :]
        img_input = img_input[r_y:r_y+H, r_x:r_x+W, :]
        # Random LR flip
        if np.random.random() < 0.5:
            img = np.copy(img[:, ::-1, :])
            img_input = np.copy(img_input[:, ::-1, :])
        train_labels[i, :, :, :] = _rgb2ycbcr(img)[:, :, 0:1]
        train_images[i, :, :, :] = _rgb2ycbcr(img_input)[:, :, 0:1]
    train_labels = train_labels / 255.0
    train_images = train_images / 255.0
    num_training_samples = train_images.shape[0]
    
    return train_images, train_labels, num_training_samples

train_images, train_labels, num_training_samples = PrepareTrainImages()

# Test images
test_filenames = glob.glob('./Set5/*.bmp')
num_test_samples = len(test_filenames)

# TF saver
saver = tf.train.Saver()

# TF Session
with tf.Session() as sess:
    # Load or init variables
    if phase == 'Train':
        tf.global_variables_initializer().run()
    else:
        saver.restore(sess, './SRCNN_models/SRCNN_iter10000.ckpt')
        print("Model restored.")
    
    if phase == 'Train':
        e = 0 # epoch
        p = 0 # pointer
        lr_curr = lr_init  # learning rate
        
        # Training
        for i in range(0, 100000):
            t = time.time()
            l_total,  _= sess.run([loss, train_SRCNN], feed_dict={inputs: train_images[p:p+B], labels: train_labels[p:p+B], lr:lr_curr})
            dT = time.time() - t
            print('Epoch: {:3d} | Iter: {:4d} | Loss: {:4.3e} | dT: {:4.3f}s'.format(e, i, l_total, dT))
            
            p += B
            if p >= num_training_samples:
                e += 1
                p = 0
                train_images, train_labels, num_training_samples = PrepareTrainImages()
            
            if i % 10000 == 0:
                save_path = saver.save(sess, './SRCNN_models/SRCNN_iter'+str(i)+'.ckpt')
                print("Model saved in file: %s" % save_path)

    # Test
    for i in range(0, num_test_samples):
        # Read a test image
        img = Image.open(test_filenames[i])
        img = img.convert('RGB')
        # Bicubic down-upsampling
        w, h = img.size
        img_input = img.resize((w//r, h//r), Image.ANTIALIAS)
        img_input = img_input.resize((w, h), Image.BICUBIC)
        img = np.asarray(img)
        img = _rgb2ycbcr(img)[:, :, 0:1]
        img_input = np.asarray(img_input)
        img_input = _rgb2ycbcr(img_input)[:, :, 0:1]
        Image.fromarray(img_input[6:-6,6:-6,0].astype(np.uint8)).save(test_filenames[i]+'_input.png')
        img_input = img_input / 255.0

        t = time.time()
        output = sess.run(outputs, feed_dict={inputs: img_input[np.newaxis, ]})
        dT = time.time() - t
        res = (np.clip(output[0,:,:,0],0,1)*255).astype(np.uint8)
        print('Test image #: {:3d} | PSNR: {:3f} | dT: {:4.3f}s'.format(i, PSNR(img[6:-6, 6:-6, 0], res), dT))

        res = Image.fromarray(res)
        res.save(test_filenames[i]+'_SRCNN_x2.png')

