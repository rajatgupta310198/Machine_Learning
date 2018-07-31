import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

BASE_PATH = '/Users/bapigupta/Documents/GitHub/Machine_Learning'
X = tf.placeholder("float")

def encoder(x_data):
    W1 = tf.Variable(tf.random_normal([784, 512]))
    b1 = tf.Variable(tf.zeros([512]))
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_data, W1), b1))

    # layer_1 = tf.layers.Dense(512,activation=tf.nn.relu)
    W2 = tf.Variable(tf.random_normal([512, 256]))
    b2 = tf.Variable(tf.zeros([256]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W2), b2))
    

    W3 = tf.Variable(tf.random_normal([256, 128]))
    b3 = tf.Variable(tf.zeros([128]))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W3), b3))
    # layer_3 = tf.layers.Dense(128, activation=tf.nn.relu)

    return layer_3


def decoder(encoder_input):
    # layer_1_decoder = tf.layers.Dense(256, activation=tf.nn.relu)

    # layer_2_decoder = tf.layers.Dense(512, activation=tf.nn.relu)

    # layer_3_decoder = tf.layers.Dense(784, activation=tf.nn.relu)

    W1 = tf.Variable(tf.random_normal([128, 256]))
    b1 = tf.Variable(tf.zeros([256]))
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_input, W1), b1))

    # # layer_1 = tf.layers.Dense(512,activation=tf.nn.relu)
    W2 = tf.Variable(tf.random_normal([256, 512]))
    b2 = tf.Variable(tf.zeros([512]))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W2), b2))
    

    W3 = tf.Variable(tf.random_normal([512, 784]))
    b3 = tf.Variable(tf.zeros([784]))
    layer_3_decoder = tf.nn.sigmoid(tf.matmul(layer_2, W3) + b3)

    return layer_3_decoder


encoder_op = encoder(X)

decoder_op = decoder(encoder_op)

loss_op = tf.reduce_mean(tf.square(X - decoder_op))

train_op = tf.train.RMSPropOptimizer(0.01).minimize(loss_op)

import matplotlib.pyplot as plt

mnist = input_data.read_data_sets(BASE_PATH + '/Deep Learning/Data/MNIST_data')


def train():
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(30000):
        
        batch_x, _ = mnist.train.next_batch(512)
        loss, _ = sess.run([loss_op, train_op], feed_dict={X:batch_x})
        print("Epoch {}, Loss {}".format(i, loss))

    saver.save(sess,'output/auto.ckpt')

def test():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,  os.getcwd() + '/auto.ckpt')
    print('restored')
    images = mnist.test.images
    image = images[12]
    decoded = sess.run(decoder_op, feed_dict={X:image.reshape(1, -1)})
    image = images[12].reshape(28, 28)
    
    plt.imshow(image, cmap='gray')
    plt.show()

    
    decoded = decoded.reshape(28, 28)
    plt.imshow(decoded, cmap='gray')
    plt.show()
    
    
# train()

test()

