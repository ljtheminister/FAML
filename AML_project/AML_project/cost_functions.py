import numpy as np
from numpy.linalg import norm

def squared_loss(x,y):
    return 0.5*norm(y-x)**2

def get_cost_function(func_name):
    if func_name == 'squared_loss':
	return squared_loss




