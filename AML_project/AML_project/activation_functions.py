import numpy as np
from numpy import exp

def logistic(x):
    return (1.0 + exp(-x))**-1

def logistic_gradient(x):
    s = logistic(x)
    return s*(1-s)

def linear(x, a, b):
    return a.dot(x)+b

def linear_gradient(x, a, b):
    return a

def tanh(x):
    return (exp(x) - exp(-x))/(exp(x) + exp(-x))

def tanh_gradient(x):
    h = tanh(x)
    return 1 - h**2

def softmax(x, i):
    n, p = x.shape
    denominator = np.zeros(p)
    for j in xrange(n):
	denominator += exp(x[j,:])
    return exp(x[i,:])/denominator, exp(x[i,:]), denominator

def softmax_gradient(x, i):
    sm, numerator, denominator = softmax(x, i) 
    return numerator*(denominator - numerator)/(denominator**2)

def get_activation_function(func_name):
    if func_name == 'logistic':
	return logistic, logistic_gradient
    elif func_name == 'tanh':
	return tanh, tanh_gradient
    elif func_name == 'softmax':
	return softmax, softmax_gradient
    elif func_name == 'linear':
	return linear, linear_gradient
