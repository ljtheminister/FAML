import pandas as pd
import numpy as np
import cPickle as pickle
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.hmm import MultinomialHMM
from sklearn.hmm import GaussianHMM
from sklearn.hmm import GMMHMM

import matplotlib
import matplotlib.pyplot as plt


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
    y_train = y[train_idx]
    X_test = X[test_idx,:]
    y_test = y[test_idx]
    return X_train, y_train, X_test, y_test, train_idx, test_idx

# Linear Regression (LR)
#def linear_regression(X_train, y_train, X_test, y_test):


#data1 = pd.read_pickle('data.pkl')
data1 = pd.read_pickle('data_norm.pkl')
y = data1['Steam'].values
#data1.drop(['WindGustM', 'WindChillM', 'HeatIndexM', 'PrecipM'], axis=1, inplace=True)
#data = (data1 - data1.mean())/data1.std()
y = data1['Steam'].values
X = np.array(data1.drop('Steam', axis=1))
X_train, y_train, X_test, y_test, train_idx, test_idx = split_train_test(X,y,.80)

LR = LinearRegression()
LR.fit(X_train,y_train)
y_LR= LR.predict(X_test)
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_SVR = svr.predict(X_test)

RF = RandomForestRegressor(n_estimators=50)
RF.fit(X_train, y_train)
y_RF= RF.predict(X_test)

y_test += 1e-10
dates = matplotlib.dates.date2num(data1.index[test_idx])

svr_error = y_test - y_SVR
rf_error = y_test - y_RF
lr_error = y_test - y_LR

plt.plot_date(dates, svr_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam per hour)')
plt.title('Absolute error of steam demand prediction (SVR)')
plt.savefig('SVR_error.png')

plt.plot_date(dates, rf_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam per hour)')
plt.title('Absolute error of steam demand prediction (RF w/ 50 trees)')
plt.savefig('RF_error.png')

plt.plot_date(dates, lr_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam per hour)')
plt.title('Absolute error of steam demand prediction (LR)')
plt.savefig('LR_error.png')

svr_rel = (svr_error)/y_test
rf_rel = (rf_error)/y_test
lr_rel = (lr_error)/y_test

plt.plot_date(dates, svr_rel)
plt.xlabel('Date')
plt.ylabel('Relative Error (Million pounds of steam per hour)')
plt.title('Relative error of steam demand prediction (SVR)')
plt.savefig('SVR_error_rel.png')

plt.plot_date(dates, rf_rel)
plt.xlabel('Date')
plt.ylabel('Relative error (Million pounds of steam per hour)')
plt.title('Relative error of steam demand prediction (RF w/ 50 trees)')
plt.savefig('RF_error_rel.png')

plt.plot_date(dates, lr_rel)
plt.xlabel('Date')
plt.ylabel('Relative error (Million pounds of steam per hour)')
plt.title('Relative error of steam demand prediction (LR)')
plt.savefig('LR_error_rel.png')

'''

# Gaussian Mixture Model Hidden Markov Models (GMM-HMM)
def gmm_hmm(y_train, y_test):
    GMM_HMM = GMMHMM(n_components=2)

'''

