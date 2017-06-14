import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('air.csv')


X = df.drop(['scaled_sound'],axis=1).as_matrix().astype(np.float32)
Y = df['scaled_sound'].as_matrix().reshape(-1,1)
X -= np.mean(X,axis=0)
X /= np.std(X,axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train.shape[0])
def neural_net_model(X_data,n_dim):
	W_1 = tf.Variable(tf.random_uniform([n_dim,10]))
	b_1 = tf.Variable(tf.zeros(10))
	layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
	layer_1 = tf.nn.relu(layer_1)

	W_2 = tf.Variable(tf.random_uniform([10,5]))
	b_2 = tf.Variable(tf.zeros([5]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.relu(layer_2)

	W_O = tf.Variable(tf.random_uniform([5,1]))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_2,W_O), b_O)

	return output



xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net_model(xs,5)
cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))



def train_model(sess):
	sess.run(tf.global_variables_initializer())

	for i in range(15):
		for j in range(X_train.shape[0]):
			sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape([1,5]),ys:y_train[j]})


		print('Cost:',sess.run(cost,feed_dict={xs:X_train,ys:y_train}))

	c = sess.run(cost,feed_dict={xs:X_test,ys:y_test})
	print('MSE:',c)
	print('Orignal :',y_test[5:6])
	print('Predicted :',sess.run(output,feed_dict={xs:X_test[5:6].reshape([1,5])}))
	if str(input('Do you want to save model ?')) =='Y':
		saver = tf.train.Saver()
		saver.save(sess,'air_model.ckpt')
		return
	else:
		train_model(sess)
# Below code for training session saves model
with tf.Session() as sess:
	train_model(sess)
	print('Model Saved...')

"""
#code after training session reload model
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess,'air_model.ckpt')
	print('Orignal :',y_test[1:2])
	print('Predicted :',sess.run(output,feed_dict={xs:X_test[1:2].reshape([1,5])}))
	c = sess.run(cost,feed_dict={xs:X_test,ys:y_test})
	print('MSE:',c)
"""
