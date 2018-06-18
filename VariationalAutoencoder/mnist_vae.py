from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layers import dense
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_hid', 256, """number of hidden units""")
tf.app.flags.DEFINE_integer('n_lat', 20, """number of latent variables""")
tf.app.flags.DEFINE_integer('batch_size', 100, """mini-batch size""")
tf.app.flags.DEFINE_integer('n_steps', 5000, """number of steps to run""")
tf.app.flags.DEFINE_string('savedir', './results/vae', """directory to save model""")
tf.app.flags.DEFINE_boolean('train', True, """training (True) / testing (False)""")

if not os.path.isdir(FLAGS.savedir):
    os.makedirs(FLAGS.savedir)

# distributions
Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli
kl_divergence = tf.contrib.distributions.kl_divergence

# load mnist dataset
mnist = input_data.read_data_sets('./MNIST_data')

# settings
n_hid = FLAGS.n_hid
n_lat = FLAGS.n_lat

# input placeholder
x = tf.placeholder(tf.float32, [None, 784]) # None = batch size
# prior distribution on z
# p_z : standard normal
p_z = Normal(loc=tf.zeros((tf.shape(x)[0], n_lat)),
        scale=tf.ones((tf.shape(x)[0], n_lat)))

# fully connected layers
# x = input, 256 = output
# tf.layers.dense(x, 256, activation=tf.nn.relu)

# define encoder
hid_enc = dense(x, n_hid, activation=tf.nn.relu)
z_mu = dense(hid_enc, n_lat)
z_sigma = dense(hid_enc, n_lat, activation=tf.nn.softplus)
q_z = Normal(loc=z_mu, scale=z_sigma)

# define decoder
z = q_z.sample()
hid_dec = dense(z, n_hid, activation=tf.nn.relu)
p_x = Bernoulli(logits=dense(hid_dec, 784))

# log-likelihood criterion
loglikel = tf.reduce_mean(tf.reduce_sum(p_x.log_prob(x), 1))

# kl-divergence between q(z|x) and p(z)
kld = tf.reduce_mean(tf.reduce_sum(kl_divergence(q_z, p_z), 1))

# ELBO
elbo = loglikel - kld
optim = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optim.minimize(-elbo)

# main training loop
batch_size = FLAGS.batch_size
n_steps = FLAGS.n_steps

sess = tf.Session()
saver = tf.train.Saver()

def train():
    logfile = open(FLAGS.savedir+'/train.log', 'w')
    logfile.write('n_hid %d, n_lat %d\n' % (n_hid, n_lat))
    sess.run(tf.global_variables_initializer())
    for t in range(n_steps):
        np_x, _ = mnist.train.next_batch(batch_size)
        sess.run(train_op, {x:np_x})
        if (t+1)%100 == 0:
            np_elbo, np_ll, np_kld = sess.run([elbo, loglikel, kld], {x:np_x})
            line = 'step %d, train batch elbo %f = %f - %f' % (t+1, np_elbo, np_ll, np_kld)
            print(line)
            logfile.write(line+'\n')
    saver.save(sess, FLAGS.savedir+'/model.ckpt')
    logfile.close()

from plots import *
def test():
    saver.restore(sess, FLAGS.savedir+'/model.ckpt')
    np_x, _ = mnist.test.next_batch(100)
    I_original = gen_tile(np_x, (10,10), img_shape=(28,28), border=True)
    np_x_recon = sess.run(p_x.sample(), {x:np_x})
    I_recon = gen_tile(np_x_recon, (10,10), img_shape=(28,28), border=True)
    np_x_gen = sess.run(p_x.sample(), {x:np_x, z:np.random.normal(size=(100,n_lat))})
    I_gen = gen_tile(np_x_gen, (10,10), img_shape=(28,28), border=True)

    fig = create_fig('original')
    plt.imshow(I_original)
    fig.savefig(FLAGS.savedir+'/original.pdf', format='pdf')

    fig = create_fig('reconstructed')
    plt.imshow(I_recon)
    fig.savefig(FLAGS.savedir+'/reconstructed.png')

    fig = create_fig('generated')
    plt.imshow(I_gen)
    fig.savefig(FLAGS.savedir+'/generated.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
