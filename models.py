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
def linear_regression(X_train, y_train, X_test, y_test):
LR = LinearRegression()
LR.fit(X_train,y_train)
y_LR= LR.predict(X_test)


data = pd.read_pickle('data.pkl')
y = np.array(data['Steam'])
X = np.array(data.drop('Steam', axis=1))

X_train, y_train, X_test, y_test, train_idx, test_idx = split_train_test(X,y,.80)



y_test += 1e-10
e_LR = np.abs(y_LR - y_test)
MAPE_LR = e_LR/y_test
plt.plot_date(dates, MAPE_LR)
dates = matplotlib.dates.date2num(data.index[test_idx])
plt.plot_date(dates, e_LR)

e_SVR = np.abs(y_SVR - y_test)
e_RF = np.abs(y_RF - y_test)

svr_error = y_test - y_SVR
rf_error = y_test - y_RF
lr_error = y_test - y_LR

plt.plot_date(dates, svr_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam)')
plt.title('Absolute error of steam demand prediction (SVR)')
plt.savefig('SVR_error.png')

plt.plot_date(dates, rf_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam)')
plt.title('Absolute error of steam demand prediction (RF w/ 50 trees)')
plt.savefig('RF_error.png')

plt.plot_date(dates, lr_error)
plt.xlabel('Date')
plt.ylabel('Absolute error (Million pounds of steam)')
plt.title('Absolute error of steam demand prediction (LR)')
plt.savefig('LR_error.png')




# Support Vector Regression (SVR)
def support_vector_regression(X_train, y_train, X_test, y_test):
    
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_SVR = svr.predict(X_test)

# Random Forest Regression (RF)
def random_forest_regression(X_train, y_train, X_test, y_test, n_trees=10):
RF = RandomForestRegressor(n_estimators=50)
RF.fit(X_train, y_train)
y_RF= RF.predict(X_test)




# Hidden Markov Models (HMM)


def hmm_multinomial(y_train, y_test):
    HMM = MultinomialHMM()

# Gaussian Mixture Model Hidden Markov Models (GMM-HMM)
def gmm_hmm(y_train, y_test):
    GMM_HMM = GMMHMM(n_components=2)



