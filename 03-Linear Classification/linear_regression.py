#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:55:17 2019

@author: pramote
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


class LinearRegression:
    def __init__(self, fit_intercept=True, normalize=False,
                 max_iter=1000, learning_rate=0.001):
        self.fit_intercept = fit_intercept # model จะมี  beta , seta = 0  มั้ย ตามสมการ y = a0(ตัวนี้) + a1x1 + a2x2 + ... +anxn
        self.normalize = normalize # จะเอา data ของแต่ละ feature ไป normalize มั้ย
        self.max_iter = max_iter # จำนวนรอบการทำงาน
        self.learning_rate = learning_rate # ความเร็วในการเรียนรู้ในแต่ละ epot ถ้าเยอะไปมันก็จะโดด ถ้าน้อยไปมันก็จะช้ามาก
        self.weights = None # ค่าของ a1 a2 ของ Feature (สปส)
        self.X = None # Feature class
        self.y = None # Label class
        self.trained = False # 

    def fit(self, X, y, sample_weight=None, verbose=False):
        self.X = X.copy()
        self.y = y

        if self.normalize:
            self.X = self.__normalize(self.X)
#            self.__normalize_output()

        if self.fit_intercept:
            intercept = np.ones((self.X.shape[0], 1))
            self.X = np.hstack((intercept, self.X))

        if sample_weight:
            self.weights = sample_weight
        else:
            self.weights = np.random.rand(self.X.shape[1])

        for iteration in range(self.max_iter):
            self.__update_weights_gd()

            # Check the cost for every 100 (for example) iterations
            if verbose and iteration % 100 == 0:
                c = self.__compute_cost()
                print('Cost:', c)
        self.trained = True

    def predict(self, X):
        if self.normalize:
            X = self.__normalize(X)

        if X.shape[1] == self.weights.shape[0] - 1:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        pred = np.dot(X, self.weights)
#        if self.normalize:
#            pred = self.__denormalize_output(pred)
        return pred

    def score(self, X, y):
        p = self.predict(X)
        return np.mean((p - y)**2)

    def get_params(self):
        return self.weights

    def __update_weights_gd(self):
        predictions = np.dot(self.X, self.weights)
        error = predictions - self.y
        delta_weights = np.dot(error, self.X)

        m = self.y.shape[0]
        self.weights -= self.learning_rate / m * delta_weights

    def __compute_cost(self):
        predictions = np.dot(self.X, self.weights)
        error = predictions - self.y

        m = self.y.shape[0]
        cost = np.dot(error.T, error)/(2*m)
        return cost

    def __normalize(self, X):
        if not self.trained:
            self.x_min = np.amin(X, axis=0)
            self.x_max = np.amax(X, axis=0)
            self.x_mean = np.mean(X, axis=0)
        return (X - self.x_mean)/(self.x_max - self.x_min)

    def __normalize_output(self):
        self.y_min = np.amin(self.y, axis=0)
        self.y_max = np.amax(self.y, axis=0)
        self.y_mean = np.mean(self.y)
        self.y = (self.y - self.y_mean)/(self.y_max - self.y_min)

    def __denormalize_output(self, y):
        return y*(self.y_max - self.y_min) + self.y_mean


def main():
    data = np.genfromtxt('weight_height_male.csv', delimiter=',')

    height = data[:, 0] # : = เอาทุก row ขอ col 0
    weight = data[:, 1] # : = เอาทุก row ขอ col 1

    X = height.reshape(height.shape[0], 1)
    y = weight
    lr = LinearRegression(normalize=True, max_iter=100000)
    lr.fit(X, y, verbose=True)
    print('Score:', lr.score(X, y))
    print('Weights:', lr.get_params())

    X_test = X
    sns.set()
    df = pd.DataFrame(data)
    df.columns = ['height', 'weight']
    df = df.set_index('height')
    df['Regression'] = lr.predict(X_test)
    sns.scatterplot(data=df)
    plt.show()

    boston = load_boston()
    X = boston['data']
    y = boston['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=101)
    lr = LinearRegression(normalize=True, max_iter=100000,
                          learning_rate=0.001)
    lr.fit(X_train, y_train, verbose=True)

    print('MSE:', lr.score(X_test, y_test))
    print('Weights:', lr.get_params())


if __name__ == "__main__":
    main()
