import tensorflow as tf
import numpy as np
import pandas as pd
features_V= []
labels_V = []
df = pd.read_csv('b_cancer.csv')
print(df.keys())
print(df.info())


X = df.drop(['diagnosis'],axis=1)
features_V = X.as_matrix().astype(np.float32)
Y = (df.loc[:,'diagnosis']=='M').astype(int)
labels_V = np.eye(2)[Y]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(features_V,labels_V,test_size=0.3)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

def neural_net_model(features_V,dim):

    W_1 = tf.Variable(tf.random_uniform([dim,10],-0.001,0.001))
    b_1 = tf.Variable(tf.zeros([10]))
    l_1 = tf.matmul(features_V,W_1) + b_1
    l_1 = tf.nn.relu(l_1)

    W_2 = tf.Variable(tf.random_uniform([10,10],-0.01,0.01))
    b_2 = tf.Variable(tf.zeros([10]))
    l_2 = tf.matmul(l_1,W_2) + b_2
    l_2 = tf.nn.relu(l_2)

    W_O = tf.Variable(tf.random_uniform([10,2],-0.1,0.1))
    b_O = tf.Variable(tf.zeros([2]))
    output = tf.matmul(l_2,W_O) + b_O
    output = tf.nn.softmax(output)

    return output

xs = tf.placeholder('float')
ys = tf.placeholder('float')

output = neural_net_model(features_V=xs,dim=30)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=ys))
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:

    for i in range(15):
        sess.run(tf.initialize_all_variables())
        for j in range(1000):
            sess.run([cost,train],feed_dict={xs:X_train,ys:y_train})
            


    correct_pred = tf.equal(tf.argmax(output,1),tf.argmax(ys,1))
    print(correct_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print(sess.run(accuracy,feed_dict={xs:X_train,ys:y_train}))

    test_acc = sess.run(accuracy,feed_dict={xs:X_test,ys:y_test})
    print(test_acc)
