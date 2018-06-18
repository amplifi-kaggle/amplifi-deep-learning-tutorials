import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Read Dataset
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Training Parameters
TOTAL_STEPS = 100000
BATCH_SIZE = 128
Z_DIM = 100
X_DIM = mnist.train.images.shape[1]
H_DIM = 128

# Creates Result Images Plot
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

# One way of defining custom initialization
def xavier_init(shape, uniform=True):
    in_dim = shape[0] # number of input nodes
    out_dim = shape[1] # number of output nodes
    if uniform is True:
        # Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of 
        # training deep feedforward neural networks." PMLR. 2010.
        dist_range = tf.sqrt(6.0 / (in_dim + out_dim))
        return tf.random_uniform(shape, minval=-dist_range, maxval=dist_range)
    else:
        stddev = tf.sqrt(3.0 / (in_dim + out_dim))
        # 3.0 is used b/c with truncated normal, values whose magnitude is more 
        # than 2 standard deviations from the mean are dropped and re-picked. 
        return tf.truncated_normal(shape, stddev=stddev)

""" Discriminator Net model """
# Discriminator Weights and Biases
x = ## CODE HERE
# Using our custom initialization
D_W1 = ## CODE HERE
D_b1 = tf.Variable(tf.zeros(shape=[H_DIM]))
# Or just use pre-defined tf initialization
D_W2 = tf.get_variable("D_W2", shape=[H_DIM, 1],
           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

D_W = [D_W1,D_W2]   # List of Discriminator Weights
D_b = [D_b1,D_b2]   # List of Discriminator Biases

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Discriminator
def discriminator(input, weights, bias):
    a1 = tf.add(tf.matmul(input, weights[0]), bias[0])
    h1 = ## CODE HERE # relu this time
    a2 = tf.add(tf.matmul(h1, weights[1]), bias[1])
    prob = tf.nn.sigmoid(a2)

    return prob, a2

""" Generator Net model """
# Generator Weights and Biases
z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
G_W1 = tf.Variable(xavier_init([Z_DIM, H_DIM], uniform=False))
G_b1 = tf.Variable(tf.zeros(shape=[H_DIM]))
G_W2 = tf.Variable(xavier_init([H_DIM, X_DIM], uniform=False))
G_b2 = ## CODE HERE

G_W = [G_W1,G_W2]   # List of Generator Weights
G_b = [G_b1,G_b2]   # List of Generator Biases

theta_G = [G_W1, G_W2, G_b1, G_b2]

# Generator
def generator(input, weights, bias):
    a1 = tf.add(tf.matmul(input, weights[0]), bias[0])
    h1 = ## CODE HERE # relu this time
    a2 = tf.add(tf.matmul(h1, weights[1]), bias[1])
    prob = tf.nn.sigmoid(a2)

    return prob

# Samples random noise
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# Losses for Generator and Discriminator
D_real, D_real_logit = discriminator(x, D_W, D_b)
D_fake, D_fake_logit = discriminator(generator(z, G_W, G_b), D_W, D_b)

D_real_loss = ## CODE HERE
D_fake_loss = ## CODE HERE
D_loss = ## CODE HERE
G_loss = ## CODE HERE

# Optimizers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    for step in range(TOTAL_STEPS): # Starts from 0
        if step % 1000 == 0:
            samples = sess.run(generator(z, G_W, G_b), feed_dict={z: sample_Z(16, Z_DIM)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_batch, _ = mnist.train.next_batch(BATCH_SIZE) # feed this batch to D

        _, D_loss_curr = ## CODE HERE
        _, G_loss_curr = ## CODE HERE

        if step % 1000 == 0:
            print('Iter: {}, D_loss: {:.5}, G_loss: {:.5}'.format(step, D_loss_curr, G_loss_curr))
