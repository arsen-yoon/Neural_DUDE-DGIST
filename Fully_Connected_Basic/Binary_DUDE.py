import numpy as np
from numpy import *
import math

def make_binary_image(im):
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    return im_bin

def bsc(x,delta):
    z=np.zeros(len(x),dtype=np.int)
    for i in range(len(x)):
        noise=int(np.random.random()<delta)
        z[i]=bit_xor(x[i],noise)
    return z

def bit_xor(a,b):
    return int(bool(a)^bool(b))

def error_rate(a,b):
    error=np.zeros(len(a))
    for i in range(len(a)):
        error[i]=bit_xor(a[i],b[i])
    return np.sum(error)/len(a)

######## 1-D DUDE ########
def One_DUDE(z,k,delta):
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)
    s_hat=x_hat.copy()

    th_0=2*delta*(1-delta)
    th_1=delta**2+(1-delta)**2
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
            m[context_str]=np.zeros(2,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
        
        if ratio < th_0:
            s_hat[i]=1
        elif ratio >= th_1:
            s_hat[i]=2
        else:
            s_hat[i]=0

    return s_hat, m


######## 1-D DUDE & 1-D N-DUDE Denoising ########
def denoise_with_s(z,s,k):
    n=len(z)
    x_hat=z.copy()
    for i in range(k,n-k):
        if s[i]==0:
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat

######## 2-D DUDE ########
def Two_DUDE(z_two, k, delta, n, offset):
    width=int(math.sqrt(n))
    th_0=2*delta*(1-delta)
    th_1=delta**2+(1-delta)**2
    s_hat=np.zeros((width,width),dtype=np.int)
    m={}
    if k==1:
        for i in range(0,width):
            for j in range(1,width-1):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        
        for i in range(0,width):
            for j in range(1,width-1):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
    elif 2<=k<=4:
        for i in range(1,width-1):
            for j in range(1,width-1):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(1,width-1):
            for j in range(1,width-1):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif k==5:
        for i in range(1,width-1):
            for j in range(2,width-2):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(1,width-1):
            for j in range(2,width-2):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif 6<=k<=12:
        for i in range(2,width-2):
            for j in range(2,width-2):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
                    
        for i in range(2,width-2):
            for j in range(2,width-2):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
    
    elif k==13:
        for i in range(2,width-2):
            for j in range(3,width-3):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(2,width-2):
            for j in range(3,width-3):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif 14<=k<=24:
        for i in range(3,width-3):
            for j in range(3,width-3):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(3,width-3):
            for j in range(3,width-3):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif k==25:
        for i in range(3,width-3):
            for j in range(4,width-4):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(3,width-3):
            for j in range(4,width-4):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif 26<=k<=40:
        for i in range(4,width-4):
            for j in range(4,width-4):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype='int')
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(4,width-4):
            for j in range(4,width-4):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif k==41:
        for i in range(4,width-4):
            for j in range(5,width-5):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(4,width-4):
            for j in range(5,width-5):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
    elif 42<=k<=60:
        for i in range(5,width-5):
            for j in range(5,width-5):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(5,width-5):
            for j in range(5,width-5):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
    elif k==61:
        for i in range(5,width-5):
            for j in range(6,width-6):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(5,width-5):
            for j in range(6,width-6):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
    elif 62<=k<=84:
        for i in range(6,width-6):
            for j in range(6,width-6):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(6,width-6):
            for j in range(6,width-6):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif k==85:
        for i in range(6,width-6):
            for j in range(7,width-7):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(6,width-6):
            for j in range(7,width-7):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    elif 86<=k<=112:
        for i in range(7,width-7):
            for j in range(7,width-7):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                if not m.has_key(context_str):
                    m[context_str]=np.zeros(2,dtype=np.int)
                    m[context_str][z_two[i,j]]=1
                else:
                    m[context_str][z_two[i,j]]+=1
        for i in range(7,width-7):
            for j in range(7,width-7):
                context=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    context=context+[z_two[a_i,a_j],z_two[b_i,b_j]]
                context_str = ''.join(str(e) for e in context)
                ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
                if ratio < th_0:
                    s_hat[i,j]=1
                elif ratio >= th_1:
                    s_hat[i,j]=2
                else:
                    s_hat[i,j]=0
        s_hat=np.reshape(s_hat,(n,))
        return s_hat,m
    
######## 2-D DUDE Denoising ########
def denoise_with_s_Two_DUDE(z_two,s,k):
    width=int(math.sqrt(z_two.shape[0]*z_two.shape[1]))
    x_hat=z_two.copy()
    s=s.reshape((z_two.shape[0], z_two.shape[1]))
    if k==1:
        for i in range(0,width):
            for j in range(1,width-1):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 2<=k<=4:
        for i in range(1,width-1):
            for j in range(1,width-1):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==5:
        for i in range(1,width-1):
            for j in range(2,width-2):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 6<=k<=12:
        for i in range(2,width-2):
            for j in range(2,width-2):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==13:
        for i in range(2,width-2):
            for j in range(3,width-3):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat    
    elif 14<=k<=24:
        for i in range(3,width-3):
            for j in range(3,width-3):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==25:
        for i in range(3,width-3):
            for j in range(4,width-4):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 26<=k<=40:
        for i in range(4,width-4):
            for j in range(4,width-4):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==41:
        for i in range(4,width-4):
            for j in range(5,width-5):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 42<=k<=60:
        for i in range(5,width-5):
            for j in range(5,width-5):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==61:
        for i in range(5,width-5):
            for j in range(6,width-6):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 62<=k<=84:
        for i in range(6,width-6):
            for j in range(6,width-6):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif k==85:
        for i in range(6,width-6):
            for j in range(7,width-7):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat
    elif 86<=k<=112:
        for i in range(7,width-7):
            for j in range(7,width-7):
                if s[i,j]==0:
                    x_hat[i,j]=z_two[i,j]
                elif s[i,j]==1:
                    x_hat[i,j]=0
                else:
                    x_hat[i,j]=1
        x_hat=x_hat.reshape(z_two.shape[0]*z_two.shape[1],)
        return x_hat