# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:55:57 2017

@author: HRDC511-07
"""

import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt



diabetes = datasets.load_diabetes()

print("data:" , diabetes.data.shape)
print(diabetes.data)
print(diabetes.target.shape)
print(diabetes.target)


#c=  zip( list(a.flatten())  , list(b.flatten()) )
#d = np.random.choice()


#test_idx = np.random.choice(c , 20)



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


a = np.linspace(-5, 5, 100)
b = np.sin(a) + np.random.randn(1, 100) * 0.2

a = a.reshape((100, 1))
b = b.reshape((100, 1))

#plt.figure()
#plt.plot(a, b, 'o')
#plt.show()

c = np.hstack((a, b))
d = np.random.choice(c.shape[0], 20)

e = []
f = []
for ind in range(0, c.shape[0]):
    if ind not in d:
        e.append(c[ind, :])
    else:
        f.append(c[ind, :])
training = np.array(e)
testing = np.array(f)
print(training)
print(testing)

learning_rate = 0.1
training_epochs = 200
display_step = 50

lambda_hyp = 1.0

X = tf.placeholder('float')
Y = tf.placeholder('float')
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

pred = tf.add(tf.multiply(X,W), b)
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/2/80
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(0, training_epochs):
    for elem in training:
        sess.run(optimizer, feed_dict={X:elem[0], Y:elem[1]})
    print(sess.run(loss, feed_dict={X:training[:, 0], Y:training[:, 1]}))

plt.plot(training[:,0], sess.run(W)*training+sess.run(b), 'o')

#plt.plot(training_data[:,0], training_data[:,1], 'o')
#plt.plot(training_data[:,0],sess.run(W)*training_data[:,0]+sess.run(b))
plt.show()











