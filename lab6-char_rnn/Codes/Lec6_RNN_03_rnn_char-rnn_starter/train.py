'''
MANY TO MANY RNN EXAMPLE
'''

from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

import sys

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    args = parser.parse_args()
    train(args)


def train(args):
    # Step 1: load data
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    # print (sys.version)
    # print (data_loader.vocab_size)
    # ^^ 65

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    # Step 2: define a model
    model = Model(args)

    # Step 3: define an optimizer
    # YOUR CODE HERE

    # Step 4: train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            # ^^ always starts with zero-filled state tensor

            for b in range(data_loader.num_batches):
                start = time.time()

                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                
                # copy the state of previous batch
                # YOUR CODE HERE
                
                train_loss, state, _ = sess.run([model.cost, model.final_state, train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))

                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
