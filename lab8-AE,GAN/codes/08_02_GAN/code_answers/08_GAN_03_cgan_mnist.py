import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Read Dataset
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Parameters
TOTAL_STEPS = 100000
BATCH_SIZE = 64
Z_DIM = 100
X_DIM = mnist.train.images.shape[1]
Y_DIM = mnist.train.labels.shape[1]
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


""" Discriminator Net model """
# Discriminator Weights and Biases
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, Y_DIM])

D_W1 = tf.get_variable("D_W1", shape=[X_DIM + Y_DIM, H_DIM],
           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b1 = tf.Variable(tf.zeros(shape=[H_DIM]))
D_W2 = tf.get_variable("D_W2", shape=[H_DIM, 1],
           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

D_W = [D_W1,D_W2]   # List of Discriminator Weights
D_b = [D_b1,D_b2]   # List of Discriminator Biases

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Discriminator 
def discriminator(x, y, weights, bias):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, weights[0]) + bias[0])
    D_logit = tf.matmul(D_h1, weights[1]) + bias[1]
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


""" Generator Net model """
# Generator Weights and Biases
z = tf.placeholder(tf.float32, shape=[None, Z_DIM])

G_W1 = tf.get_variable("G_W1", shape=[Z_DIM + Y_DIM, H_DIM],
           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b1 = tf.Variable(tf.zeros(shape=[H_DIM]))
G_W2 = tf.get_variable("G_W2", shape=[H_DIM, X_DIM],
           initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b2 = tf.Variable(tf.zeros(shape=[X_DIM]))

G_W = [G_W1,G_W2]   # List of Generator Weights
G_b = [G_b1,G_b2]   # List of Generator Biases

theta_G = [G_W1, G_W2, G_b1, G_b2]

# Generator
def generator(z, y, weights, bias):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, weights[0]) + bias[0])
    G_log_prob = tf.matmul(G_h1, weights[1]) + bias[1]
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

# Samples random noise
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# Losses for Generator and Discriminator
G_sample = generator(z, y, G_W, G_b)
D_real, D_real_logit = discriminator(x, y, D_W, D_b)
D_fake, D_fake_logit = discriminator(G_sample, y, D_W, D_b)

D_real_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_logit, labels=tf.ones_like(D_real_logit)
        )
    )
D_fake_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)
        )
    )
D_loss = D_real_loss + D_fake_loss
G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)
        )
    )

# Defining Optimizers
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    for step in range(TOTAL_STEPS):
        if step % 1000 == 0:
            n_sample = 16

            Z_sample = sample_Z(n_sample, Z_DIM)
            y_sample = np.zeros(shape=[n_sample, Y_DIM])
            # Generate '7'
            y_sample[:, 7] = 1

            samples = sess.run(G_sample, feed_dict={z: Z_sample, y:y_sample})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_batch, y_batch = mnist.train.next_batch(BATCH_SIZE) # feed this batches to D and G

        Z_sample = sample_Z(BATCH_SIZE, Z_DIM)
        _, D_loss_curr = sess.run(
            [D_solver, D_loss], 
            feed_dict={x: X_batch, z: Z_sample, y:y_batch}
            )
        _, G_loss_curr = sess.run(
            [G_solver, G_loss], 
            feed_dict={z: Z_sample, y:y_batch}
            )

        if step % 1000 == 0:
            print('Iter: {}, D_loss: {:.5}, G_loss: {:.5}'.format(step, D_loss_curr, G_loss_curr))
