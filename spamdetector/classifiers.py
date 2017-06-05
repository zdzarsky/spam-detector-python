import sys
import os
import time
import numpy as np
from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def logistic_classifier(training_x, training_y, test):
    model = None
    kf = KFold(len(training_x), n_folds=300)
    accuracy = []
    for i, test_i in kf:
        x_train = np.asarray(training_x)[i]
        y_train = np.asarray(training_y)[i]
        x_test = np.asarray(test)[test_i]
        y_test = np.asarray(training_y)[test_i]
        vectorizer = CountVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 1),
                                     stop_words='english')
        trained_x = vectorizer.fit_transform(x_train)
        trained_test = vectorizer.transform(x_test)
        model = LogisticRegression(C=1.0)
        model.fit(x_train, y_train)
    return model


def svm_classifier_loo(training_x, training_y):
    model = None
    loo = LeaveOneOut(len(training_x))
    for i, test_i in loo:
        x_train = np.asarray(training_x)[i]
        y_train = np.asarray(training_y)[i]
        model = Pipeline([('Countvector',
                           CountVectorizer(ngram_range=(1, 1),
                                           stop_words='english')),
                          ('Chi Square', SelectKBest('chi2', k=15)),
                          ('svm', SVC(C=3.0))])
        model.fit(x_train, y_train)
    return model


def svm_classifier_kfold(training_x, training_y):
    model = None
    kf = KFold(len(training_x), n_folds=300)
    for i, test_i in kf:
        x_train = np.asarray(training_x)[i]
        y_train = np.asarray(training_y)[i]
        model = Pipeline([('Countvector',
                           CountVectorizer(ngram_range=(1, 1),
                                           stop_words='english')),
                          ('Chi Square', SelectKBest('chi2', k=15)),
                          ('svm', SVC(C=3.0))])
        model.fit(x_train, y_train)
    return model
