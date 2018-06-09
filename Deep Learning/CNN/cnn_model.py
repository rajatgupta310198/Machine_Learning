import numpy as np
import os, sys
# path = os.getcwd()
# print(path)
# from image import *
# images, labels = extract_images_labels('data_batch_1')
# os.chdir(path)


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



class CNN:

    def __init__(self, learning_rate, epochs, image_h=None labels=None, mnist=True):
        """
        image_h : Image height in dataset
        Image width in dataset, this must be equal to image_h
        If not then use resize_method in this class pass images. 
        """
        self.images = []
        self.labels = labels
        self.mnist = mnist

        if self.mnist:
            self.data = input_data.read_data_sets(os.getcwd() + "/MNIST_data", one_hot=True)
            self.image_h = 28
            self.image_w = 28
            self.channels = 1
            self.check_point = 'mnist_clf.ckpt'

        else:
            
            self.image_h = image_h
            self.image_w = image_h
            self.channels = 3
            self.check_point = 'clf.ckpt'
            


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.xs = tf.placeholder("float")
        self.ys = tf.placeholder("float")
        self.rate = tf.placeholder("float")
        self.sess = tf.Session()
        

        


    def build_model(self, x):

        X_reshaped = tf.reshape(x, [-1, self.image_h, self.image_w, 1])

        conv1 = self.conv2d(X_reshaped, filters=32)
        pool1 = self.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        conv2 = self.conv2d(pool1, filters=64)
        pool2 = self.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(inputs=dense, rate=self.rate)

        logits = tf.layers.dense(dense, units=10)
        
        prediction = tf.nn.softmax(logits, name='prediction_tensor')
        return prediction, logits


    def __softmax_cross_entropy__with_logits(self, logits):

        return tf.nn.softmax_cross_entropy_with_logits(labels=self.ys, logits=logits)

    def __train(self, loss):

        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def __predict(self, x):

        return tf.argmax(x, 1)

    def train(self):
        
        predictions, logits = self.build_model(self.xs)
        loss = tf.reduce_mean(self.__softmax_cross_entropy__with_logits(logits))
        # correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.ys))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Look for another method to find accuracy
        train = self.__train(loss)

        self.sess.run(tf.global_variables_initializer())

        if self.mnist:
            for epoch in range(230):
                for _ in range(self.data.train.num_examples // 128):
                    batch_x, batch_y = self.data.train.next_batch(128)

                    l, t = self.sess.run([loss, train], feed_dict={self.xs:batch_x, self.ys:batch_y, self.rate:0.5})

                    print("Epoch {} and loss : {}".format(epoch, l))

        else:
            for epoch in range(self.epochs):
                last_index = 0
                loss_this_epoch = 0
                for _ in range(len(self.images)//64):
                    X, Y = self.images[last_index:last_index + 64], self.labels[last_index:last_index+64]
                    loss_, train_ = self.sess.run([loss, train], feed_dict={self.xs:X, self.ys:Y, self.rate:0.5})
                    last_index = last_index + 64
                    loss_this_epoch = loss_
                print("Epoch : {} , loss : {} ".format(epoch, loss_this_epoch))

                    


    def resize_image(self, images):
        """
        Pass images for reshaping
        """
        for image in images:
            self.images.append(cv2.resize(image, (self.image_h, self.image_h)))



    def conv2d(self, x, filters, kernel_size=[5, 5], padding="same", activation=tf.nn.relu):
        """
        filters : size of filter
        kernel_size : size of kernel w h
        """
        return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)

    def max_pooling2d(self, conv, pool_size=[2, 2], strides=2):
        """
        pool_size : size of pooling filter 
        strides
        """

        return tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides)


# cnn = CNN(learning_rate=0.001, epochs=1000)
# cnn.train()


