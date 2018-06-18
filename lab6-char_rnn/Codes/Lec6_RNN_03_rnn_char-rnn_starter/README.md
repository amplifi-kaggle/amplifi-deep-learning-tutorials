char-rnn-tensorflow
===

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## Basic Usage
To train with default parameters on the tinyshakespeare corpus, run `python train.py`. To access all the parameters use `python train.py --help`.

To sample from a checkpointed model, `python test.py`.

## Acknowledgement
Its original source code is from [Sherjilozair's repository](https://github.com/sherjilozair/char-rnn-tensorflow)
