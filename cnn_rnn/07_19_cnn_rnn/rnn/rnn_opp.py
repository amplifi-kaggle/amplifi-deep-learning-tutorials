#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import opportunity

from tensorflow.examples.tutorials.mnist import input_data


def fully_connected_layer(input_tensor, out_node_num, name):
	'''
	arguments:
		node_num: the number of output node
		input_tensor: tensor which is hidden representation

	return:
		perceptron result tensor
	'''
	# 1. get feature dimension from input_tensor
	input_dim = input_tensor.shape.as_list()
	
	# 2. make variable graph and matmul graph
	W = tf.get_variable(name=name+'_weights', shape=[input_dim[1], out_node_num])
	b = tf.get_variable(name=name+'_biases', shape=[out_node_num])

	result = tf.matmul(input_tensor, W) + b

	# 3. return perceptron result
	return result


def model(inputs_ph, init_state, seq_len):
	'''
	arguments:
		inputs_ph: input placeholder

	return:
		softmax result
	'''
	with tf.variable_scope('rnn') as scope:
		# 32 sample의 sensor 값을 받는 rnn 구현
		#### YOUR CODE ####
		rc1 = init_state
		for i in range(seq_len):
                        
			# input과 state를 concatenation 해줌.
			#### YOUR CODE ####
			input_prev = tf.concat([input_ph[i], rc1], 1)

			# fully connected layer
			#### YOUR CODE ####
			rc1 = fully_connected_layer(input_prev, 128, 'rc1')

			# nonlinear activation function
			#### YOUR CODE ####
			rc1 = tf.nn.sigmoid(rc1)

			# rnn은 variable을 공유하므로 variable을 resuse할 수 있도록 하는 코드가 필요합니다.
			#### YOUR CODE ####
			if i == 0:
                                tf.get_variable_scope().reuse_variables()
			


	# fully connected layer (out_node_num = class num)
	# label 만큼의 차원을 내놓는 fully connected layer
	#### YOUR CODE ####
	logits = fully_connected_layer(rc1, 17, 'logits')

	# softmax
	#### YOUR CODE ####
	softmax_result = ?

	# return softmax_result
	return softmax_result


def objective_graph(prediction, labels):
	# average cross entropy loss를 계산
	#### YOUR CODE ####
	# mce = ?

	return mce


def accuracy_graph(prediction, labels):
	# accuracy를 계산
	#### YOUR CODE ####
	# accuracy = ?

	return accuracy


def main():
        '''
        opp = opportunity.Opportunity()
        print('train input shape: '),
	print(np.shape(opp.train_input))
	print('train label shape: '),
	print(np.shape(opp.train_label))
	print('test input shape: '),
	print(np.shape(opp.test_input))
	print('test label shape: '),
	print(np.shape(opp.test_label))
	print('label list')
	print(np.unique(opp.test_label))
	'''
	
	# input placeholder 선언
	#### YOUR CODE ####
	# inputs = ?
	inputs = tf.placeholder(tf.float32, [None, 32, 113])

	# 32 개의 input을 만들어야 하므로 30개로 split하여
	# 32 개의 input tensor 'list'로 만들어줍니다.
	#### YOUR CODE ####
	inputs_unstack = tf.unstack(inputs, axis=1)
	print(inputs_unstack)
	print(len(inputs_unstack))

	# 초기 상태값을 받을 placeholder 선언
	#### YOUR CODE ####
	init_state = tf.placeholder(tf.float32, [None, 128])
	

	# 정답값을 받을 placeholder 선언
	#### YOUR CODE ####
	labels = tf.placeholder(tf.int64, [None])
	

	# int형 label을 one_hot으로 변경
	#### YOUR CODE ####
	# labels_onehot = tf.one_hot(labels, len(np.unique(opp.test_label)))
	labels_onehot = tf.one_hot(labels, 17)

	prediction = model(inputs_
	
	'''
	prediction = model(inputs_unstack, init_state, 32)
	loss = objective_graph(prediction, labels_onehot)
	accuracy = accuracy_graph(prediction, labels_onehot)

	train_vars = tf.trainable_variables()
	for var in train_vars:
		print(var.name)

	train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	init_op = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init_op)

		max_test_accuracy = 0.0
		for step in range(50000):
			batch_xs, batch_ys = opp.train_next_batch(100)
			feed_dict = {}
			feed_dict[inputs] = batch_xs
			feed_dict[labels] = batch_ys
			init_state_value = np.zeros([len(batch_ys), 128])
			feed_dict[init_state] = init_state_value
			sess.run(train_op, feed_dict=feed_dict)
			if step % 100 == 0:
				accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
				print('%dst step    train accuracy: %f' % (step, accuracy_value))
			
			if step % 1000 == 0:
				# get test accuracy
				test_xs, test_ys = opp.test_data()
				feed_dict = {}
				feed_dict[inputs] = test_xs
				feed_dict[labels] = test_ys
				init_state_value = np.zeros([len(test_ys), 128])
				feed_dict[init_state] = init_state_value
				accuracy_value = sess.run(accuracy, feed_dict=feed_dict)
				if max_test_accuracy < accuracy_value:
					max_test_accuracy = accuracy_value

				print('%dst step    accuracy: %f,  max_accuracy: %f' % 
					(step, accuracy_value, max_test_accuracy))
	'''
if __name__ == '__main__':
	main()
