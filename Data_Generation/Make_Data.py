import numpy as np
from numpy import *

def bit_xor(a,b):
    return int(bool(a)^bool(b))

def bsc(x,delta):
    z=np.zeros(len(x),dtype=np.int)
    for i in range(len(x)):
        noise=int(np.random.random()<delta)
        z[i]=bit_xor(x[i],noise)
    return z

def make_binary_image(im):
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    return im_bin