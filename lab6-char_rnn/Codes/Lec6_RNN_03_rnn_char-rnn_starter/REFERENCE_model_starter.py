import tensorflow as tf
import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # Step 2-1: create placeholder
        # YOUR CODE HERE

        # Step 2-2: define multi-layer RNN
        # YOUR CODE HERE

        self.cell = cell = # YOUR CODE HERE
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        outputs, last_state = # YOUR CODE HERE
        outputs = tf.reshape(outputs, [-1, args.rnn_size])

        # Step 2-3: compute outputs and loss
        # YOUR CODE HERE

        self.logits = # YOUR CODE HERE
        self.probs = # YOUR CODE HERE
        loss = # YOUR CODE HERE
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
