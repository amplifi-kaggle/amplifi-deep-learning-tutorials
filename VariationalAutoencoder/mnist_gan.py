from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layers import *
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_hid', 128, """number of hidden units""")
tf.app.flags.DEFINE_integer('n_noise', 50, """number of latent variables""")
tf.app.flags.DEFINE_integer('batch_size', 100, """mini-batch size""")
tf.app.flags.DEFINE_integer('n_steps', 20000, """number of steps to run""")
tf.app.flags.DEFINE_string('savedir', './results/gan', """directory to save model""")
tf.app.flags.DEFINE_boolean('train', True, """training (True) / testing (False)""")

if not os.path.isdir(FLAGS.savedir):
    os.makedirs(FLAGS.savedir)

# distributions
Bernoulli = tf.contrib.distributions.Bernoulli

# load mnist dataset
mnist = input_data.read_data_sets('../datasets/mnist')

# settings
n_hid = FLAGS.n_hid
n_noise = FLAGS.n_noise

# generating noise
def gen_noise(batch_size):
    return np.random.uniform(-1., 1., [batch_size, n_noise])

# generator network
def generate(noise, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        hid = dense(____, ____, activation=tf.nn.relu)
        x = dense(____, ____, activation=tf.nn.sigmoid)
        return x

# discriminator network
def discriminate(x, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        hid = dense(____, ____, activation=tf.nn.relu)
        logits = dense(____, ____)
        return logits

# input placeholder (real image)
real = tf.placeholder(tf.float32, [None, ____])
# noise placeholder
noise = tf.placeholder(tf.float32, [None, ____])
# fake image
fake = generate(____)

real_logits = discriminate(____, reuse=____)
fake_logits = discriminate(____, reuse=____)

def BCE(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=ref))
    return loss

# discriminator loss
L_D = BCE(____, ____) + BCE(____, ____)
L_G = BCE(____, ____)

# placeholder for learning rate
learning_rate = tf.placeholder(tf.float32)
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dis')
train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(L_D, var_list=D_vars)
lr_D = 1e-3

G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(L_G, var_list=G_vars)
lr_G = 1e-3

# main training loop
batch_size = FLAGS.batch_size
n_steps = FLAGS.n_steps

sess = tf.Session()
saver = tf.train.Saver()

def train():
    logfile = open(FLAGS.savedir+'/train.log', 'w', 0)
    logfile.write('n_hid %d, n_noise %d\n' % (n_hid, n_noise))
    sess.run(tf.global_variables_initializer())
    for t in range(n_steps):
        np_x, _ = mnist.train.next_batch(batch_size)
        feed_dict = {real:np_x, noise:gen_noise(batch_size), learning_rate:lr_D}
        sess.run(train_D, feed_dict)
        feed_dict[learning_rate] = lr_G
        sess.run(train_G, feed_dict)
        if (t+1)%100 == 0:
            feed_dict = {real:np_x, noise:gen_noise(batch_size)}
            np_L_D, np_L_G = sess.run([L_D, L_G], feed_dict)
            line = 'step %d, train batch D loss %f, G loss %f' % (t+1, np_L_D, np_L_G)
            print(line)
            logfile.write(line+'\n')
    saver.save(sess, FLAGS.savedir+'/model.ckpt')
    logfile.close()

from plots import *
def test():
    saver.restore(sess, FLAGS.savedir+'/model.ckpt')
    np_fake = sess.run(fake, {noise:gen_noise(100)})
    I_fake = gen_tile(np_fake, (10,10), img_shape=(28,28), border=True)

    fig = create_fig('fake')
    plt.imshow(I_fake)
    fig.savefig(FLAGS.savedir+'/fake.png')

    plt.show()

def main(argv=None):
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == '__main__':
    tf.app.run()
