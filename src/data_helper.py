# -*- coding: utf-8 -*-
import pandas as pd 
import os
import numpy as np 
import matplotlib.pyplot as plt

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

data_path = os.path.join(root_path, 
    'data', 'ContentNewLinkAllSample.csv')

def train_test_split(data, ratio=0.2):
    msk = np.random.rand(len(data)) < (1 - ratio)
    train = data[msk]
    test = data[~msk]
    return train, test

def get_dataset():
    data = pd.read_csv(data_path)
    data['class'] = data['class'].astype('category').cat.codes
    train, test = train_test_split(data)
    return train, test

def plot(bagging, adaboost, cls):

    x = np.asarray([5, 10, 25])
    plt.plot(x, bagging, label='Bagging')
    plt.plot(x, adaboost, label='AdaBoost')

    plt.legend()
    plt.title(cls)
    plt.show()

if __name__ == '__main__':

    bagging_tree_f1 = np.asarray([0.902, 0.913, 0.905])
    adaboost_tree_f1 = np.asarray([0.906, 0.899, 0.902])

    bagging_svm_f1 = np.asarray([0.930, 0.932, 0.928])
    adaboost_svm_f1 = np.asarray([0.925, 0.913, 0.917])

    plot(bagging_tree_f1, adaboost_tree_f1, 'Tree')
    # plot(bagging_svm_f1, adaboost_svm_f1, 'SVM')