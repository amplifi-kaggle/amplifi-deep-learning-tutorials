"""
Simple RNN example in TensorFlow
This program tries to predict the character sequence
'hihello'
"""
import tensorflow as tf
import numpy as np

x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3
              
# EACH RNN cell input_size (5) -> output_size (5)
rnn_output_size = 5
cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_output_size)
x_data = np.array(x_one_hot, dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (outputs.eval())