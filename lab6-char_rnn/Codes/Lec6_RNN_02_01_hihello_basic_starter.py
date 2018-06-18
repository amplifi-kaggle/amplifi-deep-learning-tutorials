"""
Simple RNN example in TensorFlow
This program tries to predict the character sequence
'hihello'
"""
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

# RNN parameters
num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5 # size of output from the RNN
batch_size = 1   # one sentence
sequence_length = 6  # |hihell| == 6 and |ihello| == 6
learning_rate = 0.1


# Step 1: data creation
idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello


# Step 2: create placeholder
# YOUR CODE HERE
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

# Step 3: build a model to teach 'hihello'
# YOUR CODE HERE
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
hidden, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, initial_state = initial_state)
# print hidden.get_shape()
# (1, 6, 5)

# Step 4: define a loss
# YOUR CODE HERE
outputs = tf.reshape(hidden, [-1, sequence_length, num_classes]) # -1 is about batch size (Tensorflow automatically calculates this for us)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)

# Step 5: use Adam optimizer to minimize the loss
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

# Step 6: train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(150):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))