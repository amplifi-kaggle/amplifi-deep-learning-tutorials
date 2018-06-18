from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layers import *
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_ch', 32, """number of hidden units""")
tf.app.flags.DEFINE_integer('n_noise', 50, """number of latent variables""")
tf.app.flags.DEFINE_integer('batch_size', 100, """mini-batch size""")
tf.app.flags.DEFINE_integer('n_steps', 20000, """number of steps to run""")
tf.app.flags.DEFINE_string('savedir', './results/dcgan', """directory to save model""")
tf.app.flags.DEFINE_boolean('train', True, """training (True) / testing (False)""")

if not os.path.isdir(FLAGS.savedir):
    os.makedirs(FLAGS.savedir)

# leaky ReLU
def lrelu(x, leak=0.1):
    f1 = 0.5*(1 + leak)
    f2 = 0.5*(1 - leak)
    return f1*x + f2*abs(x)

# distributions
Bernoulli = tf.contrib.distributions.Bernoulli

# load mnist dataset
mnist = input_data.read_data_sets('../datasets/mnist')

# settings
n_ch = FLAGS.n_ch
n_noise = FLAGS.n_noise

# generating noise
def gen_noise(batch_size):
    return np.random.uniform(-1., 1., [batch_size, n_noise])

# generator network
def generate(noise, is_training, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        out = dense_bn(noise, 512, is_training, activation=tf.nn.relu)
        out = dense_bn(out, 7*7*2*n_ch, is_training, activation=tf.nn.relu)
        out = tf.reshape(out, [-1, 7, 7, 2*n_ch])
        out = deconv_bn(out, n_ch, [3, 3], is_training, strides=2, padding='SAME', activation=tf.nn.relu)
        out = deconv(out, 1, [3, 3], strides=2, padding='SAME', activation=tf.nn.sigmoid)
        out = tf.reshape(out, [-1, 784])
        return out

# discriminator network
def discriminate(x, is_training, reuse=None):
    with tf.variable_scope('dis', reuse=reuse):
        out = tf.reshape(x, [-1, 28, 28, 1])
        out = conv(out, n_ch, [3, 3], strides=2, padding='SAME', activation=lrelu)
        out = conv_bn(out, 2*n_ch, [3, 3], is_training, strides=2, padding='SAME', activation=lrelu)
        out = tf.reshape(out, [-1, 7*7*2*n_ch])
        out = dense_bn(out, 512, is_training, activation=lrelu)
        logits = dense(out, 1, activation=None)
        return logits

# training flag placeholder
is_training = tf.placeholder(tf.bool)
# input placeholder (real image)
real = tf.placeholder(tf.float32, [None, 784])
# noise placeholder
noise = tf.placeholder(tf.float32, [None, n_noise])

# fake image
fake = generate(noise, is_training)

# discriminate
real_logits = discriminate(real, is_training)
fake_logits = discriminate(fake, is_training, reuse=True)

def BCE(logits, zero_or_one):
    ref = tf.zeros_like(logits) if zero_or_one is 0 else tf.ones_like(logits)
    loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=ref))
    return loss

# discriminator loss
L_D = BCE(real_logits, 1) + BCE(fake_logits, 0)
L_G = BCE(fake_logits, 1)

# placeholder for learning rate
learning_rate = tf.placeholder(tf.float32)
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dis')
train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(L_D, var_list=D_vars)
lr_D = 1e-4

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
    logfile.write('n_ch %d, n_noise %d\n' % (n_ch, n_noise))
    sess.run(tf.global_variables_initializer())
    for t in range(n_steps):
        np_x, _ = mnist.train.next_batch(batch_size)
        feed_dict = {real:np_x, noise:gen_noise(batch_size), learning_rate:lr_D, is_training:True}
        sess.run(train_D, feed_dict)
        feed_dict[learning_rate] = lr_G
        sess.run(train_G, feed_dict)
        if (t+1)%100 == 0:
            feed_dict = {real:np_x, noise:gen_noise(batch_size), is_training:False}
            np_L_D, np_L_G = sess.run([L_D, L_G], feed_dict)
            line = 'step %d, train batch D loss %f, G loss %f' % (t+1, np_L_D, np_L_G)
            print(line)
            logfile.write(line+'\n')
    saver.save(sess, FLAGS.savedir+'/model.ckpt')
    logfile.close()

from plots import *
def test():
    saver.restore(sess, FLAGS.savedir+'/model.ckpt')
    np_fake = sess.run(fake, {noise:gen_noise(100), is_training:False})
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
