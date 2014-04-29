import numpy as np
import pandas as pd
import cPickle as pickle
import random

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import hmm
from sklearn.hmm import GMMHMM


def read_csv(filename):
    return pd.read_csv('filename')

def split_train_test(X, y, train_prop):
    assert train_prop > 0 and train_prop < 1
    N, P = X.shape
    N_train = int(train_prop*N)
    N_test = N - N_train
    row_idx = [i for i in xrange(N)]
    random.shuffle(row_idx)
    train_idx = row_idx[:N_train]
    test_idx = row_idx[N_train:]
    X_train = X[train_idx,:]
    y_train = y[train_idx,:]
    X_test = X[test_idx,:]
    y_test = y[test_idx,:]
    return X_train, y_train, X_test, y_test



# Linear Regression (LR)
def linear_regression(X_train, y_train, X_test, y_test):
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_pred = LR.predict(X_test)

# Support Vector Regression (SVR)
def support_vector_regression(X_train, y_train, X_test, y_test):
    

# Random Forest Regression (RF)
def random_forest_regression(X_train, y_train, X_test, y_test, n_trees=10):
    RF = RandomForestRegressor(n_estimators=n_trees)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)




# Hidden Markov Models (HMM)

# Gaussian Mixture Model Hidden Markov Models (GMM-HMM)
def gmm_hmm(X_train, y_train, X_test, y_test, K):
    GMM_HMM = GMMHMM(n_components=2, n_mix=




