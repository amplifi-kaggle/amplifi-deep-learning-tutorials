import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # Step 2-1: create placeholder
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        # Step 2-2: define multi-layer RNN
        cells = []
        for _ in range(args.num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                     input_keep_prob=args.input_keep_prob,
                                                     output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = tf.contrib.rnn.MultiRNNCell(cells) # stack multiple rnn layer
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        # ^^ return zero-filled state tensor

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        # ^^ embedding_lookup(params, ids) : look up ids in a list of embedding tensors.

        # vv inputs : [batch_size, sequence_size, rnn_size]
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)
        outputs = tf.reshape(outputs, [-1, args.rnn_size])

        # Step 2-3: compute outputs and loss
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.seq2seq.sequence_loss(
                tf.reshape(self.logits, [-1, args.seq_length, args.vocab_size]),
                self.targets,
                tf.ones([args.batch_size, args.seq_length]))
        self.cost = tf.reduce_mean(loss)
        self.final_state = last_state

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
