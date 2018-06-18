import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

class Autoencoder(object):
    def __init__(self, n_input, n_hidden, sparsity_level, sparse_reg, \
        transfer_function=tf.nn.sigmoid, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        # Step3-0 : Setup Sparse Parameters
        self.sparsity_level = np.repeat([sparsity_level], self.n_hidden).astype(np.float32)
        self.sparse_reg = sparse_reg

        # Step3-1 : Initialize weights
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # Step3-2 : Build model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # Encoder
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        # Decoder
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # Step3-3 : Setup cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))\
        + self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def train_forward(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(p) \
            - p * tf.log(p_hat) \
            + (1 - p) * tf.log(1 - p) \
            - (1 - p) * tf.log(1 - p_hat))

    def calc_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

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
n_hidden = 1000
sparsity_level = 0.05
sparse_reg = 0.001

# Step3 : Build Autoencoder Model
autoencoder = Autoencoder(n_input = n_input,
                          n_hidden = n_hidden,
                          sparsity_level = sparsity_level,
                          sparse_reg = sparse_reg,
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

# Step4 : Train
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.train_forward(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_cost(X_test)))

# Results : Visualization
examples_to_show = 10
reconstruction = autoencoder.reconstruct(X_test[:examples_to_show])
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(X_test[i], (28, 28)))
    a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()