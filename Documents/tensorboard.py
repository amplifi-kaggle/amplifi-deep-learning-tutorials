#http://pythonkim.tistory.com/39

import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

add = tf.add(x,y)
mul = tf.multiply(x,y)

add_hist = tf.summary.scalar("add_scalar", add)
mul_hist = tf.summary.scalar("mul_scalar", mul)

merged =tf.summary.merge_all()

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    writer = tf.summary.FileWriter('./sample', sess.graph)

    for step in range(100):
        summary = sess.run(merged, feed_dict={x:step*1.0, y:2.0})
        writer.add_summary(summary, step)



#######

##from tensorflow.examples.tutorials.mnist import input_data
##
##mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
##
##import tensorflow as tf
##
##print(tf.convert_to_tensor(mnist.train.images).get_shape())
##print(tf.convert_to_tensor(mnist.train.labels).get_shape())
##
##
##W = tf.Variable(tf.zeros([784, 10]))
##b = tf.Variable(tf.zeros([10]))
##
##x = tf.placeholder("float", [None, 784])
##y_ = tf.placeholder("float", [None, 10])
##
##y = tf.nn.softmax( tf.matmul(x,W) + b )
##
##
##
###cross_entropy = -tf.reduce_sum(y_*tf.log(y))
##cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
##
##correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
##accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
##
### choose node 
##loss_hist = tf.summary.scalar("cross_entropy", cross_entropy)
##accur_hist = tf.summary.scalar("valid_hist", accuracy)
##
### merge all
##merged =tf.summary.merge_all()
##
##
##train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
##sess = tf.Session()
##sess.run(tf.initialize_all_variables())
##
### set writer of train and test
##train_writer = tf.summary.FileWriter('./sample/train/', sess.graph)
##test_writer = tf.summary.FileWriter('./sample/test/', sess.graph)
##
##for step in range(1000):
##    batch_xs, batch_ys = mnist.train.next_batch(100)
##
##    #run loss_hist and train_step
##    summary, _= sess.run([loss_hist, train_step], feed_dict = {x:batch_xs, y_:batch_ys})
##
##    #add summary of loss hist
##    train_writer.add_summary(summary, step)
##
##    #run accur_hist and accuracy
##    summary, accuracy_ = sess.run([accur_hist, accuracy], feed_dict={x:mnist.test.images,
##                                        y_:mnist.test.labels})
##    #add summary of accuracy hist
##    test_writer.add_summary(summary, step)
##
##    print('step: {:01d} | accuracy : {:.4f}'.format(step,float(accuracy_)))
##    

        




