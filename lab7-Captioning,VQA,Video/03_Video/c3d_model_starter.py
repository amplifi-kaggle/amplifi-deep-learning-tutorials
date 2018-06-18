# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""

import tensorflow as tf

# The UCF-101 dataset has 101 classes
NUM_CLASSES = 101

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

"-----------------------------------------------------------------------------------------------------------------------"

def conv3d(name, l_input, weights, bias):
  return ## CODE HERE

def max_pool(name, l_input, k):
  return ## CODE HERE

def inference_c3d(_X, _dropout, batch_size, _weights, _biases):

  # Convolution Layer
  conv1 = ## CODE HERE
  conv1 = ## CODE HERE
  pool1 = ## CODE HERE

  # Convolution Layer
  conv2 = ## CODE HERE
  conv2 = ## CODE HERE
  pool2 = ## CODE HERE

  # Convolution Layer
  conv3 = ## CODE HERE
  conv3 = ## CODE HERE
  conv3 = ## CODE HERE
  conv3 = ## CODE HERE
  pool3 = ## CODE HERE

  # Convolution Layer
  conv4 = ## CODE HERE
  conv4 = ## CODE HERE
  conv4 = ## CODE HERE
  conv4 = ## CODE HERE
  pool4 = ## CODE HERE

  # Convolution Layer
  conv5 = ## CODE HERE
  conv5 = ## CODE HERE
  conv5 = ## CODE HERE
  conv5 = ## CODE HERE
  pool5 = ## CODE HERE

  # Fully connected layer
  pool5 = ## CODE HERE
  dense1 = ## CODE HERE # Reshape conv3 output to fit dense layer input
  dense1 = ## CODE HERE

  dense1 = ## CODE HERE # Relu activation
  dense1 = ## CODE HERE

  dense2 = ## CODE HERE # Relu activation
  dense2 = ## CODE HERE

  # Output: class prediction
  out = ## CODE HERE

  return out
