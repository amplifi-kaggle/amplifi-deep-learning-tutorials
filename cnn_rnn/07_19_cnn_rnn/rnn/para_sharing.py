import tensorflow as tf

def fully_connected_layer_v(input_tensor, out_node_num):
	'''
	arguments:
		node_num: the number of output node
		input_tensor: tensor which is hidden representation

	return:
		perceptron result tensor
	'''
	# 1. get feature dimension from input_tensor
	input_dim = input_tensor.shape.as_list()
	with tf.variable_scope('fc') as scope:
		# 2. make variable graph and matmul graph
		W = tf.Variable(tf.zeros([input_dim[1], out_node_num]))
		b = tf.Variable(tf.zeros([out_node_num]))

		result = tf.matmul(input_tensor, W) + b

	# 3. return perceptron result
	return result

def fully_connected_layer_gv(input_tensor, out_node_num, name):
	'''
	arguments:
		node_num: the number of output node
		input_tensor: tensor which is hidden representation

	return:
		perceptron result tensor
	'''
	# 1. get feature dimension from input_tensor
	input_dim = input_tensor.shape.as_list()
	with tf.variable_scope('rfc') as scope:
		# 2. make variable graph and matmul graph
		W = tf.get_variable(name=name+'_weights', shape=[input_dim[1], out_node_num])
		b = tf.get_variable(name=name+'_biases', shape=[out_node_num])

		result = tf.matmul(input_tensor, W) + b

	# 3. return perceptron result
	return result


def graph1():
	with tf.Session() as sess:
		with tf.variable_scope('rnn') as scope:
			inputs = tf.placeholder(tf.float32, [None, 128])
			fc = fully_connected_layer_v(inputs, 128)
			for i in range(10):
				fc = fully_connected_layer_v(fc, 128)
		file_writer = tf.summary.FileWriter('logs', sess.graph)

		train_vars = tf.trainable_variables()
		for var in train_vars:
			print(var.name)

def graph2():
	with tf.Session() as sess:
		inputs = tf.placeholder(tf.float32, [None, 128])
		fc = inputs
		for i in range(10):
			fc = fully_connected_layer_gv(fc, 128, name='test')

			if i==0:
				tf.get_variable_scope().reuse_variables()

		file_writer = tf.summary.FileWriter('logs', sess.graph)

		train_vars = tf.trainable_variables()
		for var in train_vars:
			print(var.name)


if __name__ == '__main__':
	graph1()