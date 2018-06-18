#-*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
	# shape에 해당하는 weight를 return하는 코드 작성
	# tf.truncated_normal() 사용.
	#### YOUR CODE ####

def bias_variable(shape):
	# shape에 해당하는 bias를 return하는 코드 작성
	# tf.constant() 혹은 tf.zeros() 사용
	#### YOUR CODE ####

def conv2d(inputs, filter_size, stride, out_channel):
	# input shape를 구함 (inputs.get_shape().as_list() 사용)
	#### YOUR CODE ####

	# 위에서 정의한 weight_variable을 이용하여 convolution layer를 구성할 weight parameter를 구함
	#### YOUR CODE ####

	# 위에서 정의한 bias_variable을 이용하여 convolution layer를 구성할 bias를 parameter를 구함
	#### YOUR CODE ####

	# convolution 연산
	#### YOUR CODE ####

	# convolution 결과를 return
	#### YOUR CODE ####

def avg_pool_3x3(inputs):
	# 3x3 filter로 pooling하되, activation map size를 1/4로 줄이기위한
	# 2x2 stride를 하는 pooling 구현
	#### YOUR CODE ####

def residual_block(inputs, channel, first_layer=False):
	# residual block group의 첫 번째 layer일 경우
	# residual connection을 통해 들어오는 input map size를 1/4로 줄임. (pooling 이용)
	# 기존 네트워크처럼 일반적인 connection을 통해 들어가는 input은 convolution layer의 stride를 조절하여
	# map size를 1/4로 줄임.
	#### YOUR CODE ####

	# relu function
	#### YOUR CODE ####

	# 두 번째 convolution fucntion
	#### YOUR CODE ####

	# identity connection (simple addition)
	#### YOUR CODE ####

	# relu후 return result
	#### YOUR CODE ####

def global_avg_pooling(inputs):
	# spatial한 feature를 하나의 scalar로 만듬 (channel별 map에 대해 평균을 구함)
	#### YOUR CODE ####

def fully_connected(inputs, out_dim):
	input_shape = inputs.get_shape().as_list()
	in_channel = input_shape[1]

	weights = weight_variable([in_channel, out_dim])
	biases = bias_variable([out_dim])

	return tf.matmul(inputs, weights) + biases

def main():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, shape=[None, 784])

	# vector data를 28x28 image로 reshape
	#### YOUR CODE ####
	# x_image = ?

	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	# 첫 번째 group의 첫 번째 residual block
	#### YOUR CODE ####
	# for문을 이용하여 residual block 쌓음
	#### YOUR CODE ####
	
	# 두 번째 block group의 첫 번째 residual block
	#### YOUR CODE ####
	# for 문을 이용하여 residual block 쌓음
	#### YOUR CODE ####
	
	# 세 번째 block group의 첫 번째 residual block
	#### YOUR CODE ####
	# for 문을 이용하여 residual block 쌓음
	#### YOUR CODE ####

	# global average pooling을 이용하여 feature가 fully connected layer로 들어갈 수 있도록 vector로 만듬
	#### YOUR CODE ####

	# 10 차원의 feature를 구함
	#### YOUR CODE ####
	
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

