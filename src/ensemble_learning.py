# -*- coding: utf-8 -*-
import data_helper
import numpy as np 
from sklearn import svm
from sklearn import preprocessing
from sklearn import tree
from random import randrange

experiment_num = 5
tree_maxdepth = 5

def neg_ratio(train_label):
    '''
    0 for normal 1 for spam 
    '''
    return np.count_nonzero(train_label) / len(train_label)

def subsample(dataset, ratio=1.0):
    sample_idx = list()
    total_num = len(dataset)
    n_sample = round(total_num * ratio)
    while len(sample_idx) < n_sample:
        index = randrange(total_num)
        sample_idx.append(index)
    return dataset[sample_idx]

def preprocess(train, test):
    train_x, train_y = train[:, 0:-1], train[:, -1].squeeze().astype(np.int32)
    test_x, test_y = test[:, 0:-1], test[:, -1].squeeze().astype(np.int32)
    train_x, test_x = normalize(train_x, test_x)
    return train_x.tolist(), train_y.tolist(), test_x.tolist(), test_y.tolist()

def voting(pred, weights=None):
    '''
    2d matrix
    (instance, votes)
    '''
    winner = []
    if weights == None:
        for i in range(pred.shape[1]):
            counts = np.bincount(pred[:, i])
            winner.append(np.argmax(counts))
    else:
        assert pred.shape[0] == len(weights)
        for i in range(pred.shape[1]):
            pos_count = 0.
            neg_count = 0.
            for j in range(pred.shape[0]):
                pos_count += weights[j] * (1 - pred[j, i])
                neg_count += weights[j] * pred[j, i]
            if pos_count > neg_count:
                winner.append(int(0))
            else:
                winner.append(int(1))
    return winner

def bagging(train, test, replicates, cls):
    neg_r = neg_ratio(train[:, -1])
    cls_list = []
    for i in range(replicates):
        if cls == 'tree':
            clf = tree.DecisionTreeClassifier(max_depth=tree_maxdepth, class_weight={1:(1-neg_r)/neg_r})
        if cls == 'svm':
            clf = svm.SVC(class_weight={1:(1-neg_r)/neg_r})
        sample_train = subsample(train)
        train_x, train_y, test_x, test_y = preprocess(sample_train, test)
        clf.fit(train_x, train_y)
        cls_list.append(clf)
    predictions = [c.predict(test_x).astype(np.int32) for c in cls_list]
    prediction = voting(np.asarray(predictions))
    result = evaluate(prediction, test_y)
    return result

def ada_boost(train, test, iterations, cls):
    neg_r = neg_ratio(train[:, -1])
    train_x, train_y, test_x, test_y = preprocess(train, test)

    n_train, n_test = len(train_x), len(test_x)
    w = np.ones(n_train, dtype='float64')

    clf_list = []
    beta_list = []
    for i in range(iterations):
        if cls == 'tree':
            clf = tree.DecisionTreeClassifier(max_depth=tree_maxdepth, class_weight={1:(1-neg_r)/neg_r})
        if cls == 'svm':
            clf = svm.SVC(class_weight={1:(1-neg_r)/neg_r})
        clf.fit(train_x, train_y, sample_weight=w)
        pred_train_i = clf.predict(train_x).astype(np.int32)

        if i == 0:
            error_rate = sum(pred_train_i != train_y) / float(len(train_y))
        else:
            predictions = [c.predict(train_x).astype(np.int32) for c in clf_list]
            prediction = voting(np.asarray(predictions), beta_list)
            error = 0.
            for i in range(len(prediction)):
                if prediction[i] != train_y[i]:
                    error += 1
            error_rate = error / len(train_y)
        if error_rate > 0.5:
            print('Error rate > 0.5, jump out of loop. current i', i, error_rate)
            break
        beta = error_rate / (1 - error_rate)
        beta_list.append(np.log(1/beta))
        clf_list.append(clf)

        # update w
        for i in range(len(pred_train_i)):
            if pred_train_i[i] == train_y[i]:
                w[i] *= beta

        w /= sum(w)
        w *= len(w)

    predictions = [c.predict(test_x).astype(np.int32) for c in clf_list]
    prediction = voting(np.asarray(predictions), beta_list)
    result = evaluate(prediction, test_y)
    return result

def evaluate(prediction, label):
    results = {'tp':0, 'fp':0, 'tn':0, 'fn':0}

    for i in range(len(prediction)):
        if prediction[i] == 0 and label[i] == 0:
            results['tp'] += 1
        if prediction[i] == 0 and label[i] == 1:
            results['fp'] += 1
        if prediction[i] == 1 and label[i] == 0:
            results['fn'] += 1
        if prediction[i] == 1 and label[i] == 1:
            results['tn'] += 1

    acc = float(results['tp'] + results['tn']) / (results['tp'] + results['tn'] + results['fp'] + results['fn'])
    precision = float(results['tp']) / (results['tp'] + results['fp'])
    recall = float(results['tp']) / (results['tp'] + results['fn'])
    f1 = float(2 * results['tp']) / (2 * results['tp'] + results['fp'] + results['fn'])
    return (acc, precision, recall, f1)

def normalize(train, test):

    mean_value = np.mean(train, axis=0)
    std_value = np.std(train, axis=0)
    train = (train - mean_value) / std_value
    test = (test - mean_value) / std_value
    # result = preprocessing.scale(matrix)
    return train, test

if __name__ == '__main__':
    # split train, test 5 times:
    bagging_tree_results = []
    bagging_svm_results = []
    adaboost_tree_results = []
    adaboost_svm_results = []
    for model_num in [5, 10, 25]: # voting num / iteration num
        
        print('Model Number', model_num)
        for i in range(experiment_num): # split train/test several times
            # data preparation
            train, test = data_helper.get_dataset()
            train, test = train.as_matrix(), test.as_matrix()

            # bagging
            bagging_tree_results.append(bagging(train, test, model_num, cls='tree'))
            bagging_svm_results.append(bagging(train, test, model_num, cls='svm'))

            # adaboost M1
            adaboost_tree_results.append(ada_boost(train, test, model_num, cls='tree'))
            adaboost_svm_results.append(ada_boost(train, test, model_num, cls='svm'))

        print('\tBagging-Tree result', np.mean(bagging_tree_results, axis=0))
        print('\tBagging-SVM result', np.mean(bagging_svm_results, axis=0))
        print('\tAdaBoost-Tree result', np.mean(adaboost_tree_results, axis=0))
        print('\tAdaBoost-SVM result', np.mean(adaboost_svm_results, axis=0))