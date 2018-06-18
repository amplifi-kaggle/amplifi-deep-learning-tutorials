import tensorflow as tf
from tensorflow.python.training import moving_averages

dense = tf.layers.dense
conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose

def batch_norm(inputs, is_training, **kwargs):
    scope = kwargs.get('scope', None)
    reuse = kwargs.get('reuse', None)
    decay = kwargs.get('decay', 0.9)

    shape = inputs.shape
    num_outputs = shape[-1]
    with tf.variable_scope(scope, 'BN', [inputs], reuse=reuse):
        beta = tf.get_variable('beta', [num_outputs],
                initializer=tf.constant_initializer(0.),
                trainable=True)
        gamma = tf.get_variable('gamma', [num_outputs],
                initializer=tf.constant_initializer(1.),
                trainable=True)

        moving_mean = tf.get_variable('moving_mean', shape[-1:],
                initializer=tf.constant_initializer(0.),
                trainable=False)
        moving_var = tf.get_variable('moving_var', shape[-1:],
                initializer=tf.constant_initializer(1.),
                trainable=False)

        axis = list(range(shape.ndims - 1))
        def update_mean_var():
            mean, var = tf.nn.moments(inputs, axis)
            update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay)
            update_moving_var = moving_averages.assign_moving_average(
                    moving_var, var, decay)
            with tf.control_dependencies([update_moving_mean, update_moving_var]):
                return tf.identity(mean), tf.identity(var)

        mean, var = tf.cond(is_training,
                update_mean_var, lambda: (moving_mean, moving_var))
        return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

def dense_bn(inputs, num_outputs, is_training, **kwargs):
    activation = kwargs.pop('activation', None)
    out = dense(inputs, num_outputs, activation=None, **kwargs)
    out = batch_norm(out, is_training, **kwargs)
    out = out if activation is None else activation(out)
    return out

def conv_bn(inputs, num_outputs, kernel_size, is_training, **kwargs):
    activation = kwargs.pop('activation', None)
    out = conv(inputs, num_outputs, kernel_size, activation=None, **kwargs)
    out = batch_norm(out, is_training, **kwargs)
    out = out if activation is None else activation(out)
    return out

def deconv_bn(inputs, num_outputs, kernel_size, is_training, **kwargs):
    activation = kwargs.pop('activation', None)
    out = deconv(inputs, num_outputs, kernel_size, activation=None, **kwargs)
    out = batch_norm(out, is_training, **kwargs)
    out = out if activation is None else activation(out)
    return out
