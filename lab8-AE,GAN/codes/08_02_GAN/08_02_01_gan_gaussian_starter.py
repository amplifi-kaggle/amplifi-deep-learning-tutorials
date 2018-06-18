# Original: https://github.com/ashutoshkrjha/Generative-Adversarial-Networks-Tensorflow/blob/master/gan.py
import tensorflow as tf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Training Parameters
EPOCHS = 1000
BATCH_SIZE = 10
X_DIM = 1
Z_DIM = 1
H_DIM = 5 # Size of Hidden Layer
TOTAL_SAMPLE_SIZE = 1000
TOTAL_STEPS = int((TOTAL_SAMPLE_SIZE/BATCH_SIZE)*EPOCHS)

# Weight Initialization
# Discriminator Weights and Biases
x = tf.placeholder(tf.float32, shape=(None, X_DIM))
D_W1 = tf.Variable(tf.random_uniform([X_DIM, H_DIM], minval=0,maxval=1,dtype=tf.float32))
D_b1 = tf.Variable(tf.random_uniform([H_DIM], minval=0, maxval=1, dtype=tf.float32))
D_W2 = tf.Variable(tf.random_uniform([H_DIM, 1], minval=0, maxval=1, dtype=tf.float32))
D_b2 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype))
# Generator Weights and Biases
z = tf.placeholder(tf.float32, shape=(None, Z_DIM))
G_W1 = tf.Variable(tf.random_uniform([Z_DIM, H_DIM], minval=0, maxval=1, dtype=tf.float32))
G_b1 = tf.Variable(tf.random_uniform([H_DIM], minval=0, maxval=1, dtype=tf.float32))
G_W2 = tf.Variable(tf.random_uniform([H_DIM, X_DIM], minval=0, maxval=1, dtype=tf.float32))
G_b1 = tf.Variable(tf.random_uniform([X_DIM], minval=0, maxval=1, dtype=tf.float32))

G_W = [G_W1, G_W2]   # List of Generator Weights]
G_b = ## CODE HERE   # List of Generator Biases
D_W = ## CODE HERE   # List of Discriminator Weights
D_b = ## CODE HERE   # List of Discriminator Biases

theta_D = ## CODE HERE
theta_G = ## CODE HERE

# Drawing samples
mu = 5 # mean
sigma = 1 # standard deviation
X = np.random.normal(mu,sigma,(TOTAL_SAMPLE_SIZE,1))

# Drawing Plot of samples
plt.ion()
sorted_X = np.sort(np.transpose(X))
fit = stats.norm.pdf(sorted_X, mu, sigma)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(sorted_X,fit,c='b',marker='o',markersize=2)
ax1.hist(np.sort(X),normed=True) 
plt.pause(0.0001)

# Sampling Random Noise
def sample_Z(m, n):
    return np.random.uniform(0., 20., size=[m, n])

# Generator and Discriminator functions. 
# 'a_i' is pre-activation and 'h_i' is activation of ith layer
def discriminator(input, weights, bias):
    h0 = tf.to_float(input)
    a1 = ## CODE HERE
    h1 = ## CODE HERE
    a2 = ## CODE HERE
    y_hat = ## CODE HERE
    return y_hat, a2

def generator(input, weights, bias):
    h0 = tf.to_float(input)
    a1 = ## CODE HERE
    h1 = ## CODE HERE
    a2 = ## CODE HERE
    y_hat = a2 # Not taking sigmoid because output is just real valued and not squashed
    return y_hat

# Define Generator and Discriminator Losses
G_sample = generator(z, G_W, G_b)
D_real, _ = discriminator(x, D_W, D_b)
D_fake, _ = discriminator(G_sample, D_W, D_b)

# GAN Loss
D_loss = ## CODE HERE
G_loss = ## CODE HERE

# Defining Optimizer
train_step_d = ## CODE HERE
train_step_g = ## CODE HERE

# Result Plot Preparation
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# Training
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(1,TOTAL_STEPS):

        # Getting a batch of training samples
        begin_point = ((step-1)*BATCH_SIZE) % TOTAL_SAMPLE_SIZE
        end_point = begin_point + BATCH_SIZE
        x_batch = X[begin_point:end_point,:] # feed this batch to discriminator

        # Perform a training step
        _, D_loss_curr = ## CODE HERE
        _, G_loss_curr = ## CODE HERE
        if step % 1000 == 0 or step == 1:
            print('Iter: {}, D_loss: {:.5}, G_loss: {:.5}'.format(step, D_loss_curr, G_loss_curr))
            data = sess.run(generator(z,G_W,G_b), feed_dict={z: sample_Z(TOTAL_SAMPLE_SIZE, Z_DIM)}) # Getting generated distribution

            # Plotting Results
            ax2.clear()
            sorted_data = np.sort(np.transpose(data))
            fit = stats.norm.pdf(sorted_data, mu, sigma)
            ax2.plot(sorted_data,fit,c='b',marker='o',markersize=2)
            plt.pause(0.0001)
