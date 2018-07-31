import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

X = tf.placeholder("float")
BASE_PATH = '/Users/bapigupta/Documents/GitHub/Machine_Learning'  #path to repo

#encoder 
W1 = tf.Variable(tf.random_normal([784, 512]))
b1 = tf.Variable(tf.zeros([512]))
layer_1 = tf.matmul(X, W1) + b1
layer_1_activated = tf.nn.leaky_relu(layer_1, alpha=0.1)

#decoder
W2 = tf.Variable(tf.random_normal([512, 784]))
layer_output_logit = tf.matmul(layer_1_activated, W2)

loss = tf.reduce_mean(tf.square(X - layer_output_logit))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

mnist = input_data.read_data_sets(BASE_PATH + '/Deep Learning/Data/MNIST_data')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    for batch in range(mnist.train.num_examples//128):
        x, _ = mnist.train.next_batch(128)
        l, _ = sess.run([loss, train_op], feed_dict={X:x})

        print("Epoch {} , Batch #{}, Loss {}".format(i, batch, l))
