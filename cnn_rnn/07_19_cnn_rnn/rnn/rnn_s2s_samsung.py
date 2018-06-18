#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import stock_data

NODE_NUM = 512
SEQ_LEN = 30
OUT_SEQ_LEN = 30

def fully_connected_layer(input_tensor, out_node_num, name):
	'''
	arguments:
		input_tensor: 이전 layer의 output
		out_node_num: output node의 수

	return:
		fully connected layer의 output
	'''
	# 1. input tensor의 feature 차원을 받아옴.
	# hint: Tensor.shape.as_list()
	input_dim = input_tensor.shape.as_list()

	# 2. fully connected layer를 구성할 variable 선언
	W = tf.get_variable(name=name+'_weights', shape=[input_dim[1], out_node_num])
	b = tf.get_variable(name=name+'_biases', shape=[out_node_num])

	# 3. variable과 input tensor를 곱하고 bias를 더하여 fully connected layer의 결과를 냄
	result = tf.matmul(input_tensor, W) + b

	# 4. 결과를 return
	return result


def model(inputs_ph, init_state, seq_len, out_seq_len):
	'''
	arguments:
		inputs_ph: input placeholder
		init_state: 초기 상태 placeholder
		seq_len: input의 time 길이
		out_seq_len: output의 time 길이
	return:
		softmax result
	'''
	with tf.variable_scope('rnn') as scope:
		# 30일 간의 주가 움직임을 요약하는 encoder
		#### YOUR CODE ####
		for i in range(seq_len):
			# input과 state를 concatenation 해줌.
			#### YOUR CODE ####
			input_prev = tf.concat([input_ph[i], rc1], 1)

			# fully connected layer
			#### YOUR CODE ####
			rc1 = fully_connected_layer(input_prev, NODE_NUM, 'rc1')

			# nonlinear activation function
			# hyperbolic tangent를 사용하는 것이 regression에는 효과적인 것 같습니다.
			#### YOUR CODE ####
			rc1 = tf.nn.tanh(rc1)

			# 최종 output (logits)을 구할 것
			# 단 output이 다시 input으로 쓰일 것이므로 input과 같은 차원이어야 합니다.
			#### YOUR CODE ####
			logits = fully_connected_layer(rc1, 6, 'logits')

			# rnn은 variable을 공유하므로 variable을 resuse할 수 있도록 하는 코드가 필요합니다.
			#### YOUR CODE ####

			if i == 0:
                                tf.get_variable_scope().reuse_variables(0)

		# 30일 간의 주가움직임에 기반하여 미래 30일의 주가 움직임을 예측하는 decoder
		# 이제부터는 예측결과가 loss계산에 쓰이기 때문에 list에 logit을 모아야합니다.
		#### YOUR CODE ####
		logits_list = [tf.reshape(logits, (-1, 1, 6))]

		# 이미 하루는 encoder의 마지막 logits으로 예측하였으므로 29번만 루프를 돌도록 합니다.
		for i in range(out_seq_len-1):
			# input과 state를 concatenation 해줍니다.
			#### YOUR CODE ####
                        input_prev = tf.concat([logits, rc1], 1)

			# fully connected layer
			#### YOUR CODE ####
			rc1 = fully_connected_layer(input_prev, NODE_NUM, 'rc1')
			# nonlinear activation function
			#### YOUR CODE ####
			rc1 = tf.nn.tanh(rc1)

			# logits 구하기
			#### YOUR CODE ####
			logits = fully_connected_layer(rc1, 6, 'logits')

			# logits을 list에 추가
			#### YOUR CODE ####
			logits_list.append(tf.reshape(logits, (-1, 1, 6)))
			

		# 모은 logits을 concatenation
		# label과 같은 차원이 되도록 해줍니다.
		#### YOUR CODE ####
		result = tf.concat(logits_list, axis=1)

	# 결과를 return
	return result


def objective_graph(prediction, labels):
	# loss를 구해줍니다.
	# loss는 모든 예측값과 정답값의 square의 합의 평균
	lsm = tf.reduce_sum(tf.square(prediction - labels), [1, 2]) # 0 means batch

	return lsm


def read_data():
	dataset = stock_data.read_data('005930.KS.csv')
	train_input, train_label, test_input, test_label = stock_data.time_slicer(dataset)

	return train_input, train_label, test_input, test_label


def main():
	# plot을 위한 함수
	plt.ion()
	fig = plt.figure(1)

	# data를 읽어옴
	train_input, train_label, test_input, test_label = read_data()
	print('train input shape: '),
	print(np.shape(train_input))
	print('train label shape: '),
	print(np.shape(train_label))
	print('test input shape: '),
	print(np.shape(test_input))
	print('test label shape: '),
	print(np.shape(test_label))

	# input placeholder 선언
	inputs = tf.placeholder(tf.float32, [None, SEQ_LEN, 6])

	# input의 시간축에 대해 각 element의 최댓값을 구합니다.
	inputs_max = tf.reduce_max(inputs, 1, True)

	# input의 시간축에 대해 각 element의 최솟값을 구합니다.
        inputs_min = tf.reduce_min(inputs, 1, True)
	
	# 최솟값과 최댓값을 이용하여 normalization 해줍니다.
	# normalization을 해 주는 이유는 삼성전자 주가 값이 크기 때문에
	# weight가 큰 수를 만들도록 학습하는 것이 오래걸리기 때문입니다.
	#### YOUR CODE ####
	normed_inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

	# 30개의 input을 만들어야 하므로 30개로 split하여
	# 30 개의 input tensor 'list'로 만들어줍니다.
	#### YOUR CODE ####
	inputs_s = tf.unstack(tf.float32, [None, NODE_NUM])

	# 초기 상태값을 받을 placeholder 선언
	#### YOUR CODE ####
	labels = tf.placeholder(tf.float32, [None, OUT_SEQ_LEN, 6])

	# 정답값을 받을 placeholder 선언
	#### YOUR CODE ####
	
	
	# 정답값 또한 normalization을 하되 실제 주식에서는 정답값의 min, max 값을 모르므로
	# input의 min, max를 이용하여 normalization 해줍니다.
	#### YOUR CODE ####

	# model() 함수로부터 예측값을 얻어옵니다.
	prediction = model(inputs_s, init_state, SEQ_LEN, OUT_SEQ_LEN)
	# loss를 구합니다.
	loss = objective_graph(prediction, normed_labels)

	train_vars = tf.trainable_variables()
	for var in train_vars:
		print(var.name)

	train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
	init_op = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init_op)

		for step in range(100000):
			# train_input에서 random하게 batch_size만큼 뽑는 코드를 작성
			# hint: np.random.permutation 함수 이용
			shuffle_index = np.random.permutation(len(train_input))
			batch_xs = train_input[shuffle_index[:100]]
			batch_ys = train_label[shuffle_index[:100]]
			feed_dict = {}
			feed_dict[inputs] = batch_xs
			feed_dict[labels] = batch_ys
			# 초기 state 값을 생성. 모두 0이어야합니다. np.zeros 함수를 사용하시면 됩니다.
			init_state_value = np.zeros([len(batch_ys), NODE_NUM])
			feed_dict[init_state] = init_state_value
			sess.run(train_op, feed_dict=feed_dict)

			if step % 100 == 0:
				input_value = sess.run(normed_inputs, feed_dict=feed_dict)
				label_value = sess.run(normed_labels, feed_dict=feed_dict)
				pred_value = sess.run(prediction, feed_dict=feed_dict)
				loss_value = sess.run(loss, feed_dict=feed_dict)
				print('%dst step train loss value: %.4f' % (step, loss_value))
				plt.cla()
				plt.plot(range(-29, 1), input_value[0, :, 1], 'b')
				plt.plot(range(1, 31), label_value[0, :, 1], 'g')
				plt.plot(range(1, 31), pred_value[0, :, 1], 'g--')
				plt.title('train')
				plt.draw()
				plt.pause(0.0001)
				

			if step % 1000 == 0 and step > 5000:
				# get test accuracy
				feed_dict = {}
				feed_dict[inputs] = test_input
				feed_dict[labels] = test_label
				init_state_value = np.zeros([len(test_label), NODE_NUM])
				feed_dict[init_state] = init_state_value
				loss_value = sess.run(loss, feed_dict=feed_dict)
				print('test loss value: %.4f' % loss_value)

				
				input_value = sess.run(normed_inputs, feed_dict=feed_dict)
				label_value = sess.run(normed_labels, feed_dict=feed_dict)
				pred_value = sess.run(prediction, feed_dict=feed_dict)
				for i in range(len(test_label)):
					plt.cla()
					plt.plot(range(-29, 1), input_value[i, :, 1], 'b')
					plt.plot(range(1, 31), label_value[i, :, 1], 'g')
					plt.plot(range(1, 31), pred_value[i, :, 1], 'g--')
					plt.title('test')
					plt.draw()
					plt.pause(0.0001)


if __name__ == '__main__':
	main()
