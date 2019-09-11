#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 08:38:26 2019

@author: pramote
"""
import numpy as np
import csv
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
import timeit


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


def get_ad_click_data(n=100000):
    X_dict, y = [], []
    with open('click_through_rate/train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            i += 1
            y.append(int(row['click']))
            del row['click'], row['id'], row['hour'], row['device_id']
            del row['device_ip']
            X_dict.append(row)
            if i >= n:
                yield (X_dict, y)
                X_dict, y = [], []

n = 100000
X_dict_train, y_train = read_ad_click_data(n)
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

# Online learning

clf = SGDClassifier(loss='log', penalty=None, fit_intercept=True,
                    learning_rate='constant', eta0=0.01)


start_time = timeit.default_timer()

# there are 40428968 labelled samples, use the first twenty 100k samples
# for training, and the next 100k for testing
for i in range(20):
    print(f'Iteration: {i}')
    X_dict_train, y_train_every_100k = read_ad_click_data(100000, i * 100000)
    X_train_every_100k = dict_one_hot_encoder.transform(X_dict_train)
    clf.partial_fit(X_train_every_100k, y_train_every_100k, classes=[0, 1])


print(f"--- {timeit.default_timer() - start_time:.3f} seconds ---")

X_dict_test, y_test_next10k = read_ad_click_data(10000, (i + 1) * 200000)
X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)


predictions = clf.predict_proba(X_test_next10k)[:, 1]
roc = roc_auc_score(y_test_next10k, predictions)
print(f'The ROC AUC on testing set is: {roc:.3f}')
print(confusion_matrix(y_test_next10k, predictions > 0.5))
print(classification_report(y_test_next10k, predictions > 0.5))

# Online learning, read file without reopening the file

clf = SGDClassifier(loss='log', penalty=None, fit_intercept=True,
                    learning_rate='constant', eta0=0.01)


start_time = timeit.default_timer()

# there are 40428968 labelled samples, use the first twenty 100k samples
# for training, and the next 100k for testing
for i in range(20):
    print(f'Iteration: {i}')
    X_dict_train, y_train_every_100k = next(get_ad_click_data(100000))
    X_train_every_100k = dict_one_hot_encoder.transform(X_dict_train)
    clf.partial_fit(X_train_every_100k, y_train_every_100k, classes=[0, 1])


print(f"--- {timeit.default_timer() - start_time:.3f} seconds ---")

X_dict_test, y_test_next10k = next(get_ad_click_data(10000))
X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)


predictions = clf.predict_proba(X_test_next10k)[:, 1]
roc = roc_auc_score(y_test_next10k, predictions)
print(f'The ROC AUC on testing set is: {roc:.3f}')
print(confusion_matrix(y_test_next10k, predictions > 0.5))
print(classification_report(y_test_next10k, predictions > 0.5))
