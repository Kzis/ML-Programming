#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 08:42:02 2019

@author: pramote
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.feature_extraction import DictVectorizer
import timeit
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, fit_intercept=True, normalize=False,
                 max_iter=1000, learning_rate=0.001, alg='gd'):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.X = None
        self.y = None
        self.alg = alg

    def fit(self, X, y, sample_weight=None, verbose=False):
        self.X = X
        self.y = y
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            self.X = np.hstack((intercept, X))

        if sample_weight:
            self.weights = sample_weight
        else:
            self.weights = np.random.rand(self.X.shape[1])

        for iteration in range(self.max_iter):
            if self.alg == 'gd':
                self.__update_weights_gd()
                epocs = 100
            else:
                self.__update_weights_sgd()
                epocs = 2

            if verbose and iteration % epocs == 0:
                c = self.__compute_cost()
                print('Cost:', c)

    def predict(self, X):

        if X.shape[1] == self.weights.shape[0] - 1:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack((intercept, X))
        pred = self.__sigmoid(np.dot(X, self.weights))

        return pred

    def get_params(self):
        return self.weights

    def __sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def __update_weights_gd(self):
        predictions = self.__sigmoid(np.dot(self.X, self.weights))
        error = predictions - self.y
        delta_weights = np.dot(error, self.X)

        m = self.y.shape[0]
        self.weights -= self.learning_rate / m * delta_weights

    def __update_weights_sgd(self):
        for X_i, y_i in zip(self.X, self.y):
            predictions = self.__sigmoid(np.dot(X_i, self.weights))
            error = predictions - y_i
            delta_weights = np.dot(error, X_i)

            self.weights -= self.learning_rate * delta_weights

    def __compute_cost(self):
        predictions = self.__sigmoid(np.dot(self.X, self.weights))
        y = self.y
        cost = np.mean(-y * np.log(predictions) -
                       (1 - y) * np.log(1 - predictions))
        return cost


def read_ad_click_data(n, offset=0):
    X_dict, y = [], []
    with open('click_through_rate/train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i = 0
        for row in reader:
            i += 1
            y.append(int(row['click']))
            del row['click'], row['id'], row['hour'], row['device_id']
            del row['device_ip']
            X_dict.append(row)
            if i >= n:
                break
    return X_dict, y


def main():
    X_train = np.array([[6, 7],
                        [2, 4],
                        [3, 6],
                        [4, 7],
                        [1, 6],
                        [5, 2],
                        [2, 0],
                        [6, 3],
                        [4, 1],
                        [7, 2]])

    y_train = np.array([0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        1])

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    X_test = np.array([[6, 1],
                       [1, 3],
                       [3, 1],
                       [4, 5]])
    y_pred = lr.predict(X_test)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=['b']*5+['k']*5, marker='o')
    colours = ['k' if i >= 0.5 else 'b' for i in y_pred]
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='*', c=colours)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    n = 10000
    X_dict_train, y_train = read_ad_click_data(n)
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

    X_dict_test, y_test = read_ad_click_data(n, n)
    X_test = dict_one_hot_encoder.transform(X_dict_test)

    X_train_10k = X_train
    y_train_10k = np.array(y_train)

    start_time = timeit.default_timer()
    lr = LogisticRegression(max_iter=10000, learning_rate=0.01)
    lr.fit(X_train_10k, y_train_10k, verbose=True)

    print(f"--- {timeit.default_timer() - start_time:.3f} seconds ---")

    X_test_10k = X_test

    predictions = lr.predict(X_test_10k)
    roc = roc_auc_score(y_test, predictions)
    print(f"The ROC AUC on testing set is: {roc:.3f}")

    X_train_10k = X_train
    y_train_10k = np.array(y_train)

    start_time = timeit.default_timer()
    lr = LogisticRegression(max_iter=5, learning_rate=0.01, alg='sgd')
    lr.fit(X_train_10k, y_train_10k, verbose=True)

    print(f"--- {timeit.default_timer() - start_time:.3f} seconds ---")

    X_test_10k = X_test

    predictions = lr.predict(X_test_10k)
    roc = roc_auc_score(y_test, predictions)
    print(f"The ROC AUC on testing set is: {roc:.3f}")


if __name__ == "__main__":
    main()
