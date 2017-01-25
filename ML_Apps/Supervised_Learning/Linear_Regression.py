# linear regression

from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

X = np.array([1,2,3,4,5,6],dtype=np.float64)
Y = np.array([5,4,6,5,6,7],dtype=np.float64)


class LinearRegre:

    def __init__(self):
        self.m = 0
        self.b = 0
        self.regression_line = []


    def best_fit_slope_intercept(self,Xs, Ys):
        m = (mean(Xs) * mean(Ys) - mean(Xs * Ys)) / ((mean(Xs) ** 2) - mean(Xs ** 2))
        b = mean(Ys) - m * mean(Xs)
        return m, b

    def squared_error(self,Ys, y_line):
        return sum((y_line - Ys) ** 2)

    def coefficient_of_determination(self,Ys, y_line):
        y_mean_line = [mean(Ys) for y in Ys]
        squared_error_y_line = self.squared_error(Ys, y_line)
        squared_error_y_mean_line = self.squared_error(Ys, y_mean_line)
        return 1 - (squared_error_y_line) / (squared_error_y_mean_line)


    def fit(self,Xs,Ys):
        self.m, self.b = self.best_fit_slope_intercept(Xs, Ys)
        self.regression_line = [(self.m * x) + self.b for x in Xs]

    def predict(self,val):
        y = self.m*val + self.b
        return  y



clf = LinearRegre()
clf.fit(X,Y)
print(clf.predict(2.33))

plt.scatter(X,Y)
plt.scatter(2.33,clf.predict(2.33),color='r')
plt.show()