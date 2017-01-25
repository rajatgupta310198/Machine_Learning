# knn

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
df = pd.read_csv('breast-cancer-wisconsin.data')

df.replace('?',-9999,inplace=True)
df.drop(['id'],axis=1,inplace=True)
#print(df.keys())

X = np.array(df.drop(['class'],axis=1))
y = np.array(df['class'])

X_train , X_test ,y_train,y_test = train_test_split(X,y)

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
test = np.array([[2,3,4,5,6,7,8,9,1],[8,1,1,1,1,1,1,1,1]])
test = test.reshape(2,-1)
print(clf.predict(test))