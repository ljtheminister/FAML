
import numpy as np
import random
from numpy import sqrt
from activation_functions import *
from cost_functions import *
from numpy.linalg import norm

class NeuralNetwork:
    def __init__(self, X, y, layers, alpha=0.01, beta=0.0 ,test_prop=0.9, seed_parameter=1, activation_function='logistic', loss_function='squared_loss', output_function='linear'):

        self.X = X
        self.y = y
        self.N, self.P = X.shape

        self.seed = seed_parameter
        self.layers = layers

        self.N_train = np.floor(self.N*(1-test_prop))
        self.N_test = self.N - self.N_train

        row_idx = [i for i in xrange(self.N)]
        random.shuffle(row_idx)

        '''
        self.X_train = self.X[row_idx[:self.N_train],:]
        self.X_test = self.X[row_idx[self.N_train:],:]

        self.y_train = self.y[row_idx[:self.N_train],:]
        self.y_test = self.y[row_idx[self.N_train:],:]
        '''

        self.W = {} # weights
        self.b = {} # biases
        self.alpha = alpha # learning rate
        self.beta = beta #learning rate for bias

        self.activation_function, self.activation_gradient = get_activation_function(activation_function)
        self.loss_function = get_cost_function(loss_function)
        self.output_function, self.output_gradient = get_activation_function(output_function)

    def normalization(X):
        P = X.shape[1]
        for p in xrange(P):
            mean= np.mean(X[:,p])
            variance = np.variance(X[:,p])
            X[:,p]= (X[:,p] - mean)/variance
        return X

    def w_initial(self, input, output):
        return sqrt(6.0/(input+output))

    def initialize_weights(self):
        inputs = self.P
        outputs = self.layers[0]
        w = self.w_initial(inputs, outputs)
        self.W[0] = np.random.uniform(-w, w, size=(inputs, outputs))
        self.b[0] = 0

        for i in xrange(1, len(self.layers)):
            inputs = self.layers[i-1]
            outputs = self.layers[i]
            w = self.w_initial(inputs, outputs)
            self.W[i] = np.random.uniform(-w, w, size=(inputs, outputs))
            self.b[i] = 0

    def compute_loss(self, X, y):
        return self.loss_function(X,y)

    def compute_error(self, X, y):
        return y-X


    def feed_forward(self, X):
        N = X.shape[0]
        self.W_length = len(self.W)
        z = {}
        z_new = X

        for layer_idx in xrange(self.W_length):
            z[layer_idx] = z_new
            P = self.W[layer_idx].shape[1]
            z_new = np.zeros((N,P))
            for p in xrange(P):
                z_new[:, p] = self.activation_function(z[layer_idx].dot(self.W[layer_idx][:,p]) + np.ones(N)*self.b[layer_idx])
        #outer layer
        P = self.W[self.W_length-1].shape[1]
        for p in xrange(P):
            #z_new[:, p] = self.output_function(z[layer_idx], self.W[self.W_length-1][:,p], self.b[self.W_length-1])
            z_new[:, p] = z[layer_idx].dot(self.W[self.W_length-1][:,p]) + self.b[self.W_length-1]
        z[self.W_length] = z_new
        return z


    def back_propagation(self, X, y, z):

        batch_error = self.compute_error(z[self.W_length], y)

        # update step for output layer (linear update)
        if self.W_length-1 !=0:
            for i,e in enumerate(batch_error):
                delta = (e*z[self.W_length-1][i,:])
                self.W[self.W_length-1] += self.alpha*np.array([delta]).T
                self.b[self.W_length-1] += self.beta*e

                # backprop for inside layers (update with activation function)
                for layer_idx in xrange(self.W_length-2, 0, -1):
                    len_z = len(z[layer_idx][i,:])
                    len_W_col=len(z[layer_idx][i,:].dot(self.W[layer_idx]))
                    self.W[layer_idx] += self.alpha*e*z[layer_idx][i,:].reshape((len_z,1)).dot(self.activation_gradient(z[layer_idx][i,:].dot(self.W[layer_idx])).reshape((1,len_W_col)))
                    self.b[layer_idx] += self.beta*e

                len_W_col=len(z[0][i,:].dot(self.W[0]))
                len_z = len(z[0][i,:])
                self.W[0] += self.alpha*e*z[0][i,:].reshape((len_z,1)).dot(self.activation_gradient(z[0][i,:].dot(self.W[0])).reshape((1,len_W_col)))
                self.b[0] += self.beta*e
        else:
            for i,e in enumerate(batch_error):
                delta = (e*z[self.W_length-1][i,:])
                self.W[self.W_length-1] += self.alpha*np.array([delta]).T
                self.b[self.W_length-1] += self.beta*e



    #rbm training with real values input
    def RBM_positive(self, X):
        h=self.feed_forward(X)
        return h[1]

    def RBM_negative(self,h):
        N=h.shape[0]
        P = self.W[0].shape[0]
        v1 = np.zeros((N,P))
        v1 = h.dot((self.W[0]).T) + np.ones((N,P))*self.b[0]
        return v1

    def RBM_update(self, v , v1, h ,h1):
        self.W[0]+= self.alpha*( (v.T).dot(h) - (v1.T).dot(h1) )
        return self.W[0]


    #main for RBM and NN

    def mainNN(self,X,y):
        self.initialize_weights()

        #Neural network training
        for i in xrange(100):
            z = self.feed_forward(self.X)
            print 0.5*norm(z[self.W_length]-y)**2
            self.back_propagation(X, y, z)
            #print self.W

    def mainRBM(self,X):
        self.initialize_weights()

        #RBM training
        for j in xrange(10):
            h=self.RBM_positive(X)
            v1=self.RBM_negative(h)
            h1=self.RBM_positive(v1)
            wint=self.RBM_update(X,v1,h,h1)
        return wint

