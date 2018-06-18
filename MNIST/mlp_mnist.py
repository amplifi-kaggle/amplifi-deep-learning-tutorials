#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def fully_connected_layer(input_tensor, out_node_num):
	'''
	arguments:
		input_tensor: ���� layer�� output
		out_node_num: output node�� ��

	return:
		fully connected layer�� output
	'''
	# 1. input tensor�� feature ������ �޾ƿ�.
	# hint: Tensor.shape.as_list()
	#### YOUR CODE ####
    input_dim = input_tensor.shape.as_list()

	# 2. fully connected layer�� ������ variable ����
	#### YOUR CODE ####
    W = tf.Variable(tf.random_normal([input_dim[1], out_node_num]))
    b = tf.Variable(tf.zeros([out_node_num]))

	# 3. variable�� input tensor�� ���ϰ� bias�� ���Ͽ� fully connected layer�� ����� ��
	#### YOUR CODE ####
    result = tf.matmul(input_tensor, W) + b

	# 4. ����� return
	return result


def model(inputs_ph. keep_prob_ph):
	'''
	arguments:
		inputs_ph: input placeholder
		
	return:
		softmax result
	'''
	# fully connected layer
	#### YOUR CODE ####
    fc1 = fully_connected_layer(inputs_ph, 128) # 128 is hyperparameter -> you can use bayesian optimization

	# nonlinear activation function
	#### YOUR CODE ####
    sig1 = tf.nn.sigmoid(fc1)
    do1 = tf.nn.dropout(sig1, keep_prob_ph)

	# fully connected layer (out_node_num = class num)
	#### YOUR CODE ####
    logits = fully_connected_layer(sig1, 10)

	# softmax
	#### YOUR CODE ####
    softmax_result = tf.nn.softmax(logits)

	# return softmax_result
	return softmax_result


def objective_graph(prediction, labels):
	# average cross entropy loss�� ���
	#### YOUR CODE ####
    ce = tf.reduce_sum( -labels * tf.log(prediction), 1)
    mce = tf.reduce_mean(ce)
	# mce = ?
	return mce

def accuracy_graph(prediction, labels):
	# accuracy�� ���
	#### YOUR CODE ####
	# accuracy = ?
    pred = tf.argmax(prediction, 1)
    true_num = tf.argmax(labels, 1)
    correct_pred = tf.equal(pred, true_num)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
	return accuracy

def regularizer(lam):
	# ��� �н� ������ variable list�� ����
	# hint: tf.trainable_variables
	#### YOUR CODE ####
    train_vars = tf.trainable_variables()
    
	# �� ����Ʈ�� �� variable�� ���� l2 loss�� append
    reg_list = []
	#### YOUR CODE ####
    for var in train_vars:
        l2_loss = tf.nn.l2_loss(var)
        reg_list.append(l2_loss)
    

	# ����Ʈ�� �ִ� 1-order tensor���� ���� ����
	#### YOUR CODE ####
    
    

	# return lambda * l2 loss
	return lam * result

def main():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	print('train input shape: '),
	print(np.shape(mnist.train.images))
	print('train label shape: '),
	print(np.shape(mnist.train.labels))
	print('test input shape: '),
	print(np.shape(mnist.test.images))
	print('test label shape: '),
	print(np.shape(mnist.test.labels))

	# input�� �ޱ����� placeholder
	inputs = tf.placeholder(tf.float32, [None, 784])
	#### YOUR CODE ####
    # tf.placeholder(mnist.train.images, )
    labels = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
	
	prediction = model(inputs, keep_prob)
    
    reg = regularizer(0.002)
	loss = objective_graph(prediction, labels)
	accuracy = accuracy_graph(prediction, labels)

	train_op = tf.train.GradientDescentOptimizer(2.0).minimize(loss + reg)
	init_op = tf.global_variables_initializer()


	with tf.Session() as sess:
		sess.run(init_op)

		max_test_accuracy = 0.0
		for step in range(50000):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			sess.run(train_op, feed_dict={inputs:batch_xs, labels:batch_ys, keep_prob: 0.5})

			if step % 100 == 0:
				# get test accuracy
				accuracy_value = sess.run(accuracy, feed_dict={inputs:mnist.test.images, labels:mnist.test.labels, keep_prob: 1.0 })
				if max_test_accuracy < accuracy_value:
					max_test_accuracy = accuracy_value

				print('%dst step    accuracy: %f,  max_accuracy: %f' %
					(step, accuracy_value, max_test_accuracy))


if __name__ == '__main__':
	main()
