import numpy as np
import random as rd
from numpy import *

def make_new_proba_NN_DUDE(proba):
    proba_new=zeros((proba.shape[0],proba.shape[1]))
    for i in range(len(proba)):
        proba_new[i]=proba[i]+np.array([rd.uniform(-0.07,0.07),rd.uniform(-0.07,0.07),rd.uniform(-0.07,0.07)])
    return proba_new

def make_new_proba_NN_DUDE_Bind(proba):
    proba_new=zeros((proba.shape[0],proba.shape[1]))
    for i in range(len(proba)):
        proba_new[i]=proba[i]+np.array([rd.uniform(-0.06,0.06),rd.uniform(-0.06,0.06),rd.uniform(-0.06,0.06)])
        
    return proba_new

def make_new_proba_NN_DUDE_Bind_Norm(proba):
    proba_new=zeros((proba.shape[0],proba.shape[1]))
    for i in range(len(proba)):
        proba_new[i]=proba[i]+np.array([rd.uniform(-0.1,0.1),rd.uniform(-0.1,0.1),rd.uniform(-0.1,0.1)])
        
    return proba_new


def make_new_class(proba_new):
    new_class=zeros(proba_new.shape[0])
    for i in range(len(proba_new)):
        new_class[i]=np.argmax(proba_new[i])     
    return new_class

def rand_new(proba,delta):
    for i in range(len(proba)):
        if (rd.random()/10.)<=delta/10.:
            for j in range(len(proba[i])):
                if j!=np.argmax(proba[i]) and j!=np.argmin(proba[i]):
                    second_idx=j
                    temp_max=max(proba[i])
                    proba[i][np.argmax(proba[i])]=proba[i][second_idx]
                    proba[i][second_idx]=temp_max
    return proba