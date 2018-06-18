import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class VariationalAutoencoder(object):
    def __init__(self, n_input, n_hidden, n_z, \
        transfer_function=tf.nn.sigmoid, \
        optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.transfer = transfer_function

        # Step3-1 : Initialize weights
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # Step3-2 : Build model
        # Encoder
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        layer_1 = self.transfer(tf.add(tf.matmul(self.x, self.weights['encoder_w1']), 
                                           self.weights['encoder_b1']))

        # Encoder : Get z_mean, z_log_sigma
        self.z_mean = tf.add(tf.matmul(layer_1, self.weights['out_mean_w1']), self.weights['out_mean_b1'])
        self.z_log_sigma_sq = tf.add(tf.matmul(layer_1, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

        # sample from gaussian distribution
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_z]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Decoder : given z_code, generate an image
        layer_2 = self.transfer(tf.add(tf.matmul(self.z, self.weights['decoder_w2']), self.weights['decoder_b2']))
        self.reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_w3']), self.weights['decoder_b3']))



        # Step3-3 : Setup cost
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # Mathmatical expression of KL divergence with Unit Gaussian.
        latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) \
            + tf.square(self.z_log_sigma_sq) - tf.log(tf.square(self.z_log_sigma_sq)) - 1,1)  
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['encoder_w1'] = tf.get_variable("encoder_w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['encoder_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))

        all_weights['out_mean_w1'] = tf.get_variable("out_mean_w1", shape=[self.n_hidden, self.n_z],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['out_mean_b1'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))

        all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_hidden, self.n_z],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_z], dtype=tf.float32))

        all_weights['decoder_w2'] = tf.get_variable('decoder_w2',shape=[self.n_z, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['decoder_b2'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['decoder_w3'] = tf.get_variable('decoder_w3',shape=[self.n_hidden, self.n_input],
            initializer=tf.contrib.layers.xavier_initializer())
        tf.Variable(tf.zeros([self.n_z, self.n_hidden], dtype=tf.float32))
        all_weights['decoder_b3'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def train_forward(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


def min_max_scale(X_train, X_test):
    preprocessor = prep.MinMaxScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


# Step1 : Setup DATA
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
X_train, X_test = min_max_scale(mnist.train.images, mnist.test.images)


# Step2 : Setup Parameters
n_samples = int(mnist.train.num_examples)
training_epochs = 5
batch_size = 128
learning_rate = 0.001
display_step = 1

n_input = 784
n_hidden = 200
n_z = 20


# Step3 : Build Autoencoder Model
vae = VariationalAutoencoder(n_input = n_input, n_hidden = n_hidden, n_z = n_z,\
                                     transfer_function=tf.nn.sigmoid,\
                                     optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate))

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = vae.train_forward(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# Visualization : Results1
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)
plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.draw()
plt.savefig('v1_VAE_result1.png')
plt.waitforbuttonpress()


# Visualization : Results2
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()
plt.grid()
plt.tight_layout()
plt.draw()
plt.savefig('v1_VAE_result2.png')
plt.waitforbuttonpress()


# Visualization : Results3
nx = ny = 20
x_values = np.linspace(-2, 2, nx)
y_values = np.linspace(-2, 2, ny)
canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi] * 10])
        x_reconstruct = vae.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_reconstruct[0].reshape(28, 28)
plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.tight_layout()
plt.draw()
plt.savefig('v1_VAE_result3.png')
plt.waitforbuttonpress()