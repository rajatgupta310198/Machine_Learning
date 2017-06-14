import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

number_of_hidden_nodes = 250
iris = load_iris()
X = iris.data
Y = iris.target
Y = np.eye(3)[Y]

def neural_net(X,n_dim):
   W_1 = tf.Variable(tf.random_normal([n_dim,number_of_hidden_nodes]))
   b_1 = tf.Variable(tf.zeros([number_of_hidder_nodes]))
   l_1 = tf.matmul(X,W_1) + b_1
   l_1 = tf.nn.relu(l_1)

   W_O = tf.Variable(tf.random_normal([number_of_hidder_nodes,3]))
   b_O = tf.Variable(tf.zeros([3]))
   output = tf.matmul(l_1,W_O) + b_O
   #output = tf.nn.softmax(output)

   return output


xs = tf.placeholder("float")
ys = tf.placeholder("float")

output = neural_net(xs,X.shape[1])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=ys))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

sess = tf.Session()
#saver = tf.train.Saver()
#saver.restore(sess,'iris_model.ckpt')
sess.run(tf.global_variables_initializer())


correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

for i in range(100):
    for j in range(X_train.shape[0]):
        sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape([1,X_train.shape[1]]),ys:y_train[j]})


    print('Epoch :',i,'Accuracy :',sess.run(accuracy,feed_dict={xs:X_train,ys:y_train}))




print('Training Completed !')

a = sess.run(accuracy,feed_dict={xs:X_test,ys:y_test})
print('Accuracy :',a)
if a>=0.95:
    saver = tf.train.Saver()
    saver.save(sess,'iris_model.ckpt')
    print('Model Saved')
