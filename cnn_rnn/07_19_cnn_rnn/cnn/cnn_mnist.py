#-*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
	# shape에 해당하는 weight를 return하는 코드 작성
	# tf.truncated_normal() 사용.
	#### YOUR CODE ####
	#print(shape)
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	# shape에 해당하는 bias를 return하는 코드 작성
	# tf.constant() 혹은 tf.zeros() 사용
	#### YOUR CODE ####
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(inputs, filter_size, stride, out_channel):
	# input shape를 구함 (inputs.get_shape().as_list() 사용)
	#### YOUR CODE ####
	input_shape = inputs.get_shape().as_list()
	in_channel = input_shape[3]

	# 위에서 정의한 weight_variable을 이용하여 convolution layer를 구성할 weight parameter를 구함
	#### YOUR CODE ####
	weights = weight_variable([filter_size, filter_size, in_channel, out_channel])

	# 위에서 정의한 bias_variable을 이용하여 convolution layer를 구성할 bias를 parameter를 구함
	#### YOUR CODE ####
	biases = bias_variable([out_channel])

	# convolution 연산
	#### YOUR CODE ####
	# tf.nn.conv2d(inputs, weigths, filter, padding)
	conv = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding='SAME')
	

	# convolution 결과를 return
	#### YOUR CODE ####
	return conv + biases
	
	

def fully_connected(inputs, out_dim):
	input_shape = inputs.get_shape().as_list()
	in_channel = input_shape[1]

	weights = weight_variable([in_channel, out_dim])
	biases = bias_variable([out_dim])

	return tf.matmul(inputs, weights) + biases

def flatten_batch(inputs):
	# h * w의 activation map을 fully connected layer로 들어갈 수 있도록 vector로 만들어줌
	#### YOUR CODE ####
	input_shape = inputs.get_shape().as_list()
	# batch * h * w * c
	return tf.reshape(inputs, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
	

def main():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, shape=[None, 784])

	# vector data를 28x28 image로 reshape
	#### YOUR CODE ####
	# x_image = ?
	# Redefining x as 4D tensor [-1, 28, 28, 1].  The last 1 indicates the dimension of color channel
	# batch size is not defined. So initialize it to -1.
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# convolution layer
	#### YOUR CODE ####
	# input=x_image, filter=5, stride=1, outputchannel = 32
	conv1 = conv2d(x_image, 5, 1, 32)
	print(conv1.shape)
        
	# relu function
	#### YOUR CODE ####
	relu1 = tf.nn.relu(conv1)
	print(relu1.shape)
	
	# max pooling
	#### YOUR CODE ####
	pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

	# convolution layer
	#### YOUR CODE ####
	# stride = 1, outputchannel = 64
	conv2 = conv2d(pool1, 5, 1, 64)

	# relu function
	#### YOUR CODE ####
	relu2 = tf.nn.relu(conv2)

	# max pooling
	#### YOUR CODE ####
	pool2 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

	# flatten_batch()
	#### YOUR CODE ####
	flatten_b = flatten_batch(pool2)

	# fully connected layer
	#### YOUR CODE ####
	fc1 = fully_connected(flatten_b, 512)

	# relu function
	#### YOUR CODE ####
	relu3 = tf.nn.relu(fc1)

	# fully connected layer
	#### YOUR CODE ####
	logits = fully_connected(relu3, 10)
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y_: batch[1]})

if __name__ == '__main__':
	main()
