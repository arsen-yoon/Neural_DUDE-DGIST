import numpy as np
import random

from numpy import *

def rand_initialization(incre_dim, num_nodes):
    added_weights=zeros((incre_dim,num_nodes))
    for i in range(added_weights.shape[0]):
        for j in range(added_weights.shape[1]):
            r=np.random.normal(0,2/(2*2*2.))
            added_weights[i,j]=r
    return added_weights