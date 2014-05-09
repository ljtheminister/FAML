

import numpy as np

import random
from numpy import sqrt
from activation_functions import *
from cost_functions import *
from numpy.linalg import norm
from NN_Test2 import *
'''

#NN 3 - 4 - 1

N = 10 #batch size
P = 4 #input size
layers = [4,10,10, 1]
X = np.random.uniform(0, 1, size=(N,P))
y = 2*np.mean(X)
nn = NeuralNetwork(X, y, layers)
nn.mainNN(X,y)

'''

#DBN

#global input
X = np.random.uniform(0, 1, size=(10,1)) #input
y = 2*X

#RBM1 1 - 4
N = X.shape[0] #batch size
P = X.shape[1] #input size
layers1 = [4]
nn1 = NeuralNetwork(X, y, layers1)
w0=nn1.mainRBM(X)


#RBM2 4 - 6
X1 = X.dot(w0)
N = X.shape[0] #batch size
P = X.shape[1] #input size

layers2 = [6]
nn2 = NeuralNetwork(X1, y, layers2)
w1=nn2.mainRBM(X1)


#construct last layer for regression
layers3=[1]
X2=X1.dot(w1)
nn3= NeuralNetwork(X2, y, layers3)
nn3.mainNN(X2,y)

