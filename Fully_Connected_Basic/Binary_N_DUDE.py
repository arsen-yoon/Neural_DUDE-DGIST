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

######## 1-D N-DUDE ########
def make_data_for_One_NN_DUDE(Z,k,L,nb_classes,n):
    C=zeros((n-2*k, 2*k*nb_classes))
    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        C[i-k,]=c_i
        
    Y=dot(Z[k:n-k,],L)    
    return C,Y

######## 1-D N-DUDE Padding ########
def make_data_for_One_NN_DUDE_PD(P,Z,k,L,nb_classes,n):
    width=int(math.sqrt(n))
    C=zeros((n, 2*k*nb_classes))
    P_new=hstack((zeros((width,k,nb_classes)),P,zeros((width,k,nb_classes))))
    for i in range(width):
        for j in range(k,width+k):
            c_i=vstack((P_new[i,j-k:j,],P_new[i,j+1:j+k+1,])).reshape(1,2*k*nb_classes)
            C[i*width+j-k]=c_i
    Y=dot(Z,L)    
    return C,Y

######## 1-D N-DUDE Padding Bound ########
def make_data_for_One_NN_DUDE_PD_LB(P,Z,im_bin,k,L_lower,nb_classes,n):
    width=int(math.sqrt(n))
    L_split=np.split(L_lower, 2)
    L0=L_split[0]
    L1=L_split[1]
    listing = np.reshape(P, (n, nb_classes))
    Y=zeros((listing.shape[0], L_lower.shape[1]))
    
    for i in range(width):
        for j in range(k,width+k):
            if im_bin[i,j-k]==0:
                Y_i=dot(listing[i*width+j-k,:], L0)
                Y[i*width+j-k,]=Y_i
            else:
                Y_i=dot(listing[i*width+j-k,:], L1)
                Y[i*width+j-k,]=Y_i    
    return Y

######## 1-D DUDE & N-DUDE, Bound Denoising ########
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

######## 1-D N-DUDE Padding & Bound Denoising ########
def denoise_with_s_One_NN_PD(z,s):
    n=len(z)
    x_hat=z.copy()
    for i in range(n):
        if s[i]==0:
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat

######## 1-D N-DUDE Bound ########
def make_data_for_One_NN_DUDE_LB(Z,k,L,x,z,nb_classes,n):
    #C=zeros((n-2*k, 2*k*nb_classes))
    Y=zeros((n-2*k, L.shape[1]))
    L_split=np.split(L, 2)
    L0=L_split[0]
    L1=L_split[1]
    for i in range(k,n-k):
        #c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        #C[i-k,]=c_i
        if x[i]==0:
            Y_i=dot(Z[i],L0)
            Y[i-k,]=Y_i
        else:
            Y_i=dot(Z[i],L1)
            Y[i-k,]=Y_i
    return Y

######## 1-D N-DUDE Context Bind ########
def make_data_for_One_NN_DUDE_Context_Bind(z,Z,k,L,nb_classes,n):
    Context=[]
    m={}
    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        context_str=''.join(str(e) for e in c_i)
        if not m.has_key(context_str): # Check whether the context is appeared before.
            m[context_str]=np.zeros(2,dtype=np.int) # 2 is alphabet size(binary).
            m[context_str][z[i]]=1 # z[i]=0, m[context_str][0] is counted, otherwise m[context_str][1] is counted.
            Context+=c_i.tolist() # if there is a new context, add it to the list
        else:
            m[context_str][z[i]]+=1
    key=m.keys()
    Context=np.reshape(Context, (len(Context),2*k*nb_classes))
    Y=zeros((len(Context),3))
    Y_Norm=Y.copy()
    for j in range(len(Context)):
        if m.has_key(str(Context[j])):
            Y[j]=dot(m[str(Context[j])],L)
    
    for l in range(len(Context)): # for normalization, divide the num of points that have the context
        Y_Norm[l]=Y[l]/sum(m[str(Context[l])])
        
    return Context,Y,Y_Norm,key,m

######## 1-D N-DUDE Context Bind Lower Bound ########
def make_data_for_One_NN_DUDE_Context_Bind_LB(x,z,Z,k,L,nb_classes,n):
    Context=[]
    m={}
    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        context_str=''.join(str(e) for e in c_i)
        if not m.has_key(context_str):
            m[context_str]=np.zeros((2,2),dtype=np.int) # for the case x[i],z[i] = 0 or 1 respectively
            m[context_str][x[i],z[i]]=1 
            Context+=c_i.tolist()
        else:
            m[context_str][x[i],z[i]]+=1
    key=m.keys()
    Context=np.reshape(Context,(len(Context),2*k*nb_classes))
    Y=zeros((len(Context),3))
    Y_Norm=Y.copy()
    for j in range(len(Context)):
        if m.has_key(str(Context[j])):
            Y[j]=dot(np.reshape(m[str(Context[j])],(1,4)),L)
    
    for l in range(len(Context)):
        Y_Norm[l]=Y[l]/sum(m[str(Context[l])])
        
    return Context,Y,Y_Norm,key,m

######## 2-D N-DUDE ########
def make_data_for_Two_NN_DUDE(P,Z,k,L,nb_classes,n,offset):
    width=int(math.sqrt(n))
    if k==1:
        # context generation part #
        C=zeros((n-2*width, 2*k*nb_classes))
        for i in range(0,width): # i-th row
            for j in range(1,width-1): # j-th col
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[i*(width-2)+j-1,]=c_i # each row has width-2 cols
        # pseudo label generation part #
        col_cut = np.split(P, [1, width-1], axis=1)
        listing = np.reshape(col_cut[1], (col_cut[1].shape[0]*col_cut[1].shape[1], col_cut[1].shape[2]))
        Y=dot(listing,L)        
        return C,Y
    elif 2<=k<=4:
        C=zeros((n-2*width-2*(width-2), 2*k*nb_classes))
        for i in range(1,width-1): 
            for j in range(1,width-1):
                c_i=[] 
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-1)*(width-2)+j-1,]=c_i
        
        col_cut = np.split(P, [1, width-1], axis=1)
        row_cut = np.split(col_cut[1], [1, width-1])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==5:
        C=zeros((n-2*width-2*(width-2)-2*(width-2), 2*k*nb_classes))
        for i in range(1,width-1): 
            for j in range(2,width-2):
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-1)*(width-4)+j-2,]=c_i
        
        col_cut = np.split(P, [2, width-2], axis=1)
        row_cut = np.split(col_cut[1], [1, width-1])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 6<=k<=12:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4), 2*k*nb_classes))
        for i in range(2,width-2): 
            for j in range(2,width-2): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-2)*(width-4)+j-2,]=c_i

        col_cut = np.split(P, [2, width-2], axis=1)
        row_cut = np.split(col_cut[1], [2, width-2])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==13:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4), 2*k*nb_classes))
        for i in range(2,width-2): 
            for j in range(3,width-3): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-2)*(width-6)+j-3,]=c_i

        col_cut = np.split(P, [3, width-3], axis=1)
        row_cut = np.split(col_cut[1], [2, width-2])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 14<=k<=24:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6), 2*k*nb_classes))
        for i in range(3,width-3): 
            for j in range(3,width-3): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-3)*(width-6)+j-3,]=c_i

        col_cut = np.split(P, [3, width-3], axis=1)
        row_cut = np.split(col_cut[1], [3, width-3])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==25:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6), 2*k*nb_classes))
        for i in range(3,width-3): 
            for j in range(4,width-4): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-3)*(width-8)+j-4,]=c_i

        col_cut = np.split(P, [4, width-4], axis=1)
        row_cut = np.split(col_cut[1], [3, width-3])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 26<=k<=40:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8), 2*k*nb_classes))
        for i in range(4,width-4): 
            for j in range(4,width-4): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-4)*(width-8)+j-4,]=c_i

        col_cut = np.split(P, [4, width-4], axis=1)
        row_cut = np.split(col_cut[1], [4, width-4])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==41:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8),
                 2*k*nb_classes))
        for i in range(4,width-4): 
            for j in range(5,width-5): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-4)*(width-10)+j-5,]=c_i

        col_cut = np.split(P, [5, width-5], axis=1)
        row_cut = np.split(col_cut[1], [4, width-4])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 42<=k<=60:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10),2*k*nb_classes))
        for i in range(5,width-5): 
            for j in range(5,width-5): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-5)*(width-10)+j-5,]=c_i

        col_cut = np.split(P, [5, width-5], axis=1)
        row_cut = np.split(col_cut[1], [5, width-5])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==61:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10),2*k*nb_classes))
        for i in range(5,width-5): 
            for j in range(6,width-6): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-5)*(width-12)+j-6,]=c_i

        col_cut = np.split(P, [6, width-6], axis=1)
        row_cut = np.split(col_cut[1], [5, width-5])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 62<=k<=84:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12),2*k*nb_classes))
        for i in range(6,width-6): 
            for j in range(6,width-6): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-6)*(width-12)+j-6,]=c_i

        col_cut = np.split(P, [6, width-6], axis=1)
        row_cut = np.split(col_cut[1], [6, width-6])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==85:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12),2*k*nb_classes))
        for i in range(6,width-6): 
            for j in range(7,width-7): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-6)*(width-14)+j-7,]=c_i

        col_cut = np.split(P, [7, width-7], axis=1)
        row_cut = np.split(col_cut[1], [6, width-6])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 86<=k<=112:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14),2*k*nb_classes))
        for i in range(7,width-7): 
            for j in range(7,width-7): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-7)*(width-14)+j-7,]=c_i

        col_cut = np.split(P, [7, width-7], axis=1)
        row_cut = np.split(col_cut[1], [7, width-7])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==113:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14),2*k*nb_classes))
        for i in range(7,width-7): 
            for j in range(8,width-8): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-7)*(width-16)+j-8,]=c_i

        col_cut = np.split(P, [8, width-8], axis=1)
        row_cut = np.split(col_cut[1], [7, width-7])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 114<=k<=144:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16),2*k*nb_classes))
        for i in range(8,width-8): 
            for j in range(8,width-8): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-8)*(width-16)+j-8,]=c_i

        col_cut = np.split(P, [8, width-8], axis=1)
        row_cut = np.split(col_cut[1], [8, width-8])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==145:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-
                 2*(width-16),2*k*nb_classes))
        for i in range(8,width-8): 
            for j in range(9,width-9): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-8)*(width-18)+j-9,]=c_i

        col_cut = np.split(P, [9, width-9], axis=1)
        row_cut = np.split(col_cut[1], [8, width-8])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 146<=k<=180:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-
                 2*(width-16)-2*(width-18),2*k*nb_classes))
        for i in range(9,width-9): 
            for j in range(9,width-9): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-9)*(width-18)+j-9,]=c_i

        col_cut = np.split(P, [9, width-9], axis=1)
        row_cut = np.split(col_cut[1], [9, width-9])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==181:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18),2*k*nb_classes))
        for i in range(9,width-9): 
            for j in range(10,width-10): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-9)*(width-20)+j-10,]=c_i

        col_cut = np.split(P, [10, width-10], axis=1)
        row_cut = np.split(col_cut[1], [9, width-9])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 182<=k<=220:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20),2*k*nb_classes))
        for i in range(10,width-10): 
            for j in range(10,width-10): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-10)*(width-20)+j-10,]=c_i

        col_cut = np.split(P, [10, width-10], axis=1)
        row_cut = np.split(col_cut[1], [10, width-10])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==221:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20),2*k*nb_classes))
        for i in range(10,width-10): 
            for j in range(11,width-11): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-10)*(width-22)+j-11,]=c_i

        col_cut = np.split(P, [11, width-11], axis=1)
        row_cut = np.split(col_cut[1], [10, width-10])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 222<=k<=264:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22),2*k*nb_classes))
        for i in range(11,width-11): 
            for j in range(11,width-11): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-11)*(width-22)+j-11,]=c_i

        col_cut = np.split(P, [11, width-11], axis=1)
        row_cut = np.split(col_cut[1], [11, width-11])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==265:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22),2*k*nb_classes))
        for i in range(11,width-11): 
            for j in range(12,width-12): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-11)*(width-24)+j-12,]=c_i

        col_cut = np.split(P, [12, width-12], axis=1)
        row_cut = np.split(col_cut[1], [11, width-11])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 266<=k<=312:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)-2*(width-24),2*k*nb_classes))
        for i in range(12,width-12): 
            for j in range(12,width-12): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-12)*(width-24)+j-12,]=c_i

        col_cut = np.split(P, [12, width-12], axis=1)
        row_cut = np.split(col_cut[1], [12, width-12])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==313:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24),2*k*nb_classes))
        for i in range(12,width-12): 
            for j in range(13,width-13): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-12)*(width-26)+j-13,]=c_i

        col_cut = np.split(P, [13, width-13], axis=1)
        row_cut = np.split(col_cut[1], [12, width-12])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 314<=k<=364:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26),2*k*nb_classes))
        for i in range(13,width-13): 
            for j in range(13,width-13): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-13)*(width-26)+j-13,]=c_i

        col_cut = np.split(P, [13, width-13], axis=1)
        row_cut = np.split(col_cut[1], [13, width-13])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==365:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26),2*k*nb_classes))
        for i in range(13,width-13): 
            for j in range(14,width-14): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-13)*(width-28)+j-14,]=c_i

        col_cut = np.split(P, [14, width-14], axis=1)
        row_cut = np.split(col_cut[1], [13, width-13])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 366<=k<=420:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28),2*k*nb_classes))
        for i in range(14,width-14): 
            for j in range(14,width-14): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-14)*(width-28)+j-14,]=c_i

        col_cut = np.split(P, [14, width-14], axis=1)
        row_cut = np.split(col_cut[1], [14, width-14])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==421:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28),2*k*nb_classes))
        for i in range(14,width-14): 
            for j in range(15,width-15): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-14)*(width-30)+j-15,]=c_i

        col_cut = np.split(P, [15, width-15], axis=1)
        row_cut = np.split(col_cut[1], [14, width-14])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 422<=k<=480:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)-2*(width-30),2*k*nb_classes))
        for i in range(15,width-15): 
            for j in range(15,width-15): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-15)*(width-30)+j-15,]=c_i

        col_cut = np.split(P, [15, width-15], axis=1)
        row_cut = np.split(col_cut[1], [15, width-15])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==481:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30),2*k*nb_classes))
        for i in range(15,width-15): 
            for j in range(16,width-16): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-15)*(width-32)+j-16,]=c_i

        col_cut = np.split(P, [16, width-16], axis=1)
        row_cut = np.split(col_cut[1], [15, width-15])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 482<=k<=544:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32),2*k*nb_classes))
        for i in range(16,width-16): 
            for j in range(16,width-16): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-16)*(width-32)+j-16,]=c_i

        col_cut = np.split(P, [16, width-16], axis=1)
        row_cut = np.split(col_cut[1], [16, width-16])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==545:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32),2*k*nb_classes))
        for i in range(16,width-16): 
            for j in range(17,width-17): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-16)*(width-34)+j-17,]=c_i

        col_cut = np.split(P, [17, width-17], axis=1)
        row_cut = np.split(col_cut[1], [16, width-16])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 546<=k<=612:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34),2*k*nb_classes))
        for i in range(17,width-17): 
            for j in range(17,width-17): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-17)*(width-34)+j-17,]=c_i

        col_cut = np.split(P, [17, width-17], axis=1)
        row_cut = np.split(col_cut[1], [17, width-17])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==613:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34),2*k*nb_classes))
        for i in range(17,width-17): 
            for j in range(18,width-18): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-17)*(width-36)+j-18,]=c_i

        col_cut = np.split(P, [18, width-18], axis=1)
        row_cut = np.split(col_cut[1], [17, width-17])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 614<=k<=684:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)-2*(width-36),2*k*nb_classes))
        for i in range(18,width-18): 
            for j in range(18,width-18): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-18)*(width-36)+j-18,]=c_i

        col_cut = np.split(P, [18, width-18], axis=1)
        row_cut = np.split(col_cut[1], [18, width-18])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==685:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2(width-36),2*k*nb_classes))
        for i in range(18,width-18): 
            for j in range(19,width-19): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-18)*(width-38)+j-19,]=c_i

        col_cut = np.split(P, [19, width-19], axis=1)
        row_cut = np.split(col_cut[1], [18, width-18])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 686<=k<=760:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38),2*k*nb_classes))
        for i in range(19,width-19): 
            for j in range(19,width-19): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-19)*(width-38)+j-19,]=c_i

        col_cut = np.split(P, [19, width-19], axis=1)
        row_cut = np.split(col_cut[1], [19, width-19])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==761:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38),2*k*nb_classes))
        for i in range(19,width-19): 
            for j in range(20,width-20): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-19)*(width-40)+j-20,]=c_i

        col_cut = np.split(P, [20, width-20], axis=1)
        row_cut = np.split(col_cut[1], [19, width-19])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 762<=k<=840:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40),2*k*nb_classes))
        for i in range(20,width-20): 
            for j in range(20,width-20): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-20)*(width-40)+j-20,]=c_i

        col_cut = np.split(P, [20, width-20], axis=1)
        row_cut = np.split(col_cut[1], [20, width-20])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==841:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40),2*k*nb_classes))
        for i in range(20,width-20): 
            for j in range(21,width-21): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-20)*(width-42)+j-21,]=c_i

        col_cut = np.split(P, [21, width-21], axis=1)
        row_cut = np.split(col_cut[1], [20, width-20])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 842<=k<=924:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)-2*(width-42),2*k*nb_classes))
        for i in range(21,width-21): 
            for j in range(21,width-21): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-21)*(width-42)+j-21,]=c_i

        col_cut = np.split(P, [21, width-21], axis=1)
        row_cut = np.split(col_cut[1], [21, width-21])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==925:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42),2*k*nb_classes))
        for i in range(21,width-21): 
            for j in range(22,width-22): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-21)*(width-44)+j-22,]=c_i

        col_cut = np.split(P, [22, width-22], axis=1)
        row_cut = np.split(col_cut[1], [21, width-21])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 926<=k<=1012:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44),2*k*nb_classes))
        for i in range(22,width-22): 
            for j in range(22,width-22): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-22)*(width-44)+j-22,]=c_i

        col_cut = np.split(P, [22, width-22], axis=1)
        row_cut = np.split(col_cut[1], [22, width-22])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1013:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44),2*k*nb_classes))
        for i in range(22,width-22): 
            for j in range(23,width-23): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-22)*(width-46)+j-23,]=c_i

        col_cut = np.split(P, [23, width-23], axis=1)
        row_cut = np.split(col_cut[1], [22, width-22])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1014<=k<=1104:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46),2*k*nb_classes))
        for i in range(23,width-23):
            for j in range(23,width-23):
                c_i=[]
                for l in range(1,k+1):
                    
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-23)*(width-46)+j-23,]=c_i

        col_cut = np.split(P, [23, width-23], axis=1)
        row_cut = np.split(col_cut[1], [23, width-23])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1105:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46),2*k*nb_classes))
        for i in range(23,width-23): 
            for j in range(24,width-24): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-23)*(width-48)+j-24,]=c_i

        col_cut = np.split(P, [24, width-24], axis=1)
        row_cut = np.split(col_cut[1], [23, width-23])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1106<=k<=1200:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)-2*(width-48),2*k*nb_classes))
        for i in range(24,width-24): 
            for j in range(24,width-24): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-24)*(width-48)+j-24,]=c_i

        col_cut = np.split(P, [24, width-24], axis=1)
        row_cut = np.split(col_cut[1], [24, width-24])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1201:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48),2*k*nb_classes))
        for i in range(24,width-24): 
            for j in range(25,width-25): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-24)*(width-50)+j-25,]=c_i

        col_cut = np.split(P, [25, width-25], axis=1)
        row_cut = np.split(col_cut[1], [24, width-24])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1202<=k<=1300:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50),2*k*nb_classes))
        for i in range(25,width-25): 
            for j in range(25,width-25): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-25)*(width-50)+j-25,]=c_i

        col_cut = np.split(P, [25, width-25], axis=1)
        row_cut = np.split(col_cut[1], [25, width-25])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1301:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50),2*k*nb_classes))
        for i in range(25,width-25): 
            for j in range(26,width-26): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-25)*(width-52)+j-26,]=c_i

        col_cut = np.split(P, [26, width-26], axis=1)
        row_cut = np.split(col_cut[1], [25, width-25])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1302<=k<=1404:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52),2*k*nb_classes))
        for i in range(26,width-26): 
            for j in range(26,width-26): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-26)*(width-52)+j-26,]=c_i

        col_cut = np.split(P, [26, width-26], axis=1)
        row_cut = np.split(col_cut[1], [26, width-26])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1405:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52),2*k*nb_classes))
        for i in range(26,width-26): 
            for j in range(27,width-27): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-26)*(width-54)+j-27,]=c_i

        col_cut = np.split(P, [27, width-27], axis=1)
        row_cut = np.split(col_cut[1], [26, width-26])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1406<=k<=1512:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54),2*k*nb_classes))
        for i in range(27,width-27): 
            for j in range(27,width-27): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-27)*(width-54)+j-27,]=c_i

        col_cut = np.split(P, [27, width-27], axis=1)
        row_cut = np.split(col_cut[1], [27, width-27])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1513:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54),2*k*nb_classes))
        for i in range(27,width-27): 
            for j in range(28,width-28): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-27)*(width-56)+j-28,]=c_i

        col_cut = np.split(P, [28, width-28], axis=1)
        row_cut = np.split(col_cut[1], [27, width-27])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1514<=k<=1624:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56),2*k*nb_classes))
        for i in range(28,width-28): 
            for j in range(28,width-28): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-28)*(width-56)+j-28,]=c_i

        col_cut = np.split(P, [28, width-28], axis=1)
        row_cut = np.split(col_cut[1], [28, width-28])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1625:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56),2*k*nb_classes))
        for i in range(28,width-28): 
            for j in range(29,width-29): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-28)*(width-58)+j-29,]=c_i

        col_cut = np.split(P, [29, width-29], axis=1)
        row_cut = np.split(col_cut[1], [28, width-28])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1626<=k<=1740:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58),2*k*nb_classes))
        for i in range(29,width-29): 
            for j in range(29,width-29): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-29)*(width-58)+j-29,]=c_i

        col_cut = np.split(P, [29, width-29], axis=1)
        row_cut = np.split(col_cut[1], [29, width-29])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1741:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58),2*k*nb_classes))
        for i in range(29,width-29): 
            for j in range(30,width-30): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-29)*(width-60)+j-30,]=c_i

        col_cut = np.split(P, [30, width-30], axis=1)
        row_cut = np.split(col_cut[1], [29, width-29])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1742<=k<=1860:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60),2*k*nb_classes))
        for i in range(30,width-30): 
            for j in range(30,width-30): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-30)*(width-60)+j-30,]=c_i

        col_cut = np.split(P, [30, width-30], axis=1)
        row_cut = np.split(col_cut[1], [30, width-30])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1861:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60),2*k*nb_classes))
        for i in range(30,width-30): 
            for j in range(31,width-31): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-30)*(width-62)+j-31,]=c_i

        col_cut = np.split(P, [31, width-31], axis=1)
        row_cut = np.split(col_cut[1], [30, width-30])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1862<=k<=1984:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62),2*k*nb_classes))
        for i in range(31,width-31): 
            for j in range(31,width-31): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-31)*(width-62)+j-31,]=c_i

        col_cut = np.split(P, [31, width-31], axis=1)
        row_cut = np.split(col_cut[1], [31, width-31])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==1985:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62),2*k*nb_classes))
        for i in range(31,width-31): 
            for j in range(32,width-32): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-31)*(width-64)+j-32,]=c_i

        col_cut = np.split(P, [32, width-32], axis=1)
        row_cut = np.split(col_cut[1], [31, width-31])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 1986<=k<=2112:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64),2*k*nb_classes))
        for i in range(32,width-32): 
            for j in range(32,width-32): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-32)*(width-64)+j-32,]=c_i

        col_cut = np.split(P, [32, width-32], axis=1)
        row_cut = np.split(col_cut[1], [32, width-32])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2113:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64),2*k*nb_classes))
        for i in range(32,width-32): 
            for j in range(33,width-33): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-32)*(width-66)+j-33,]=c_i

        col_cut = np.split(P, [33, width-33], axis=1)
        row_cut = np.split(col_cut[1], [32, width-32])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2114<=k<=2244:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66),2*k*nb_classes))
        for i in range(33,width-33): 
            for j in range(33,width-33): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-33)*(width-66)+j-33,]=c_i

        col_cut = np.split(P, [33, width-33], axis=1)
        row_cut = np.split(col_cut[1], [33, width-33])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2245:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66),2*k*nb_classes))
        for i in range(33,width-33): 
            for j in range(34,width-34): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-33)*(width-68)+j-34,]=c_i

        col_cut = np.split(P, [34, width-34], axis=1)
        row_cut = np.split(col_cut[1], [33, width-33])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2246<=k<=2380:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68),2*k*nb_classes))
        for i in range(34,width-34): 
            for j in range(34,width-34): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-34)*(width-68)+j-34,]=c_i

        col_cut = np.split(P, [34, width-34], axis=1)
        row_cut = np.split(col_cut[1], [34, width-34])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2381:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68),2*k*nb_classes))
        for i in range(34,width-34): 
            for j in range(35,width-35): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-34)*(width-70)+j-35,]=c_i

        col_cut = np.split(P, [35, width-35], axis=1)
        row_cut = np.split(col_cut[1], [34, width-34])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2382<=k<=2520:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70),2*k*nb_classes))
        for i in range(35,width-35): 
            for j in range(35,width-35): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-35)*(width-70)+j-35,]=c_i

        col_cut = np.split(P, [35, width-35], axis=1)
        row_cut = np.split(col_cut[1], [35, width-35])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2521:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70),
                 2*k*nb_classes))
        for i in range(35,width-35): 
            for j in range(36,width-36): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-35)*(width-72)+j-36,]=c_i

        col_cut = np.split(P, [36, width-36], axis=1)
        row_cut = np.split(col_cut[1], [35, width-35])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2522<=k<=2664:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72),2*k*nb_classes))
        for i in range(36,width-36): 
            for j in range(36,width-36): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-36)*(width-72)+j-36,]=c_i

        col_cut = np.split(P, [36, width-36], axis=1)
        row_cut = np.split(col_cut[1], [36, width-36])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2665:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72),2*k*nb_classes))
        for i in range(36,width-36): 
            for j in range(37,width-37): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-36)*(width-74)+j-37,]=c_i

        col_cut = np.split(P, [37, width-37], axis=1)
        row_cut = np.split(col_cut[1], [36, width-36])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2666<=k<=2812:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72)-2*(width-74),2*k*nb_classes))
        for i in range(37,width-37): 
            for j in range(37,width-37): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-37)*(width-74)+j-37,]=c_i

        col_cut = np.split(P, [37, width-37], axis=1)
        row_cut = np.split(col_cut[1], [37, width-37])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2813:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72)-2*(width-74)-2*(width-74),2*k*nb_classes))
        for i in range(37,width-37): 
            for j in range(38,width-38): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-37)*(width-76)+j-38,]=c_i

        col_cut = np.split(P, [38, width-38], axis=1)
        row_cut = np.split(col_cut[1], [37, width-37])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2814<=k<=2964:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72)-2*(width-74)-2*(width-74)-2*(width-76),2*k*nb_classes))
        for i in range(38,width-38): 
            for j in range(38,width-38): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-38)*(width-76)+j-38,]=c_i

        col_cut = np.split(P, [38, width-38], axis=1)
        row_cut = np.split(col_cut[1], [38, width-38])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif k==2965:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72)-2*(width-74)-2*(width-74)-2*(width-76)-2*(width-76),
                 2*k*nb_classes))
        for i in range(38,width-38): 
            for j in range(39,width-39): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-38)*(width-78)+j-39,]=c_i

        col_cut = np.split(P, [39, width-39], axis=1)
        row_cut = np.split(col_cut[1], [38, width-38])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
    elif 2966<=k<=3120:
        C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
                 -2*(width-10)-2*(width-10)-2*(width-12)-2*(width-12)-2*(width-14)-2*(width-14)-2*(width-16)-2*(width-16)
                 -2*(width-18)-2*(width-18)-2*(width-20)-2*(width-20)-2*(width-22)-2*(width-22)
                 -2*(width-24)-2*(width-24)-2*(width-26)-2*(width-26)-2*(width-28)-2*(width-28)
                 -2*(width-30)-2*(width-30)-2*(width-32)-2*(width-32)-2*(width-34)-2*(width-34)
                 -2*(width-36)-2*(width-36)-2*(width-38)-2*(width-38)-2*(width-40)-2*(width-40)
                 -2*(width-42)-2*(width-42)-2*(width-44)-2*(width-44)-2*(width-46)-2*(width-46)
                 -2*(width-48)-2*(width-48)-2*(width-50)-2*(width-50)-2*(width-52)-2*(width-52)
                 -2*(width-54)-2*(width-54)-2*(width-56)-2*(width-56)-2*(width-58)-2*(width-58)
                 -2*(width-60)-2*(width-60)-2*(width-62)-2*(width-62)-2*(width-64)-2*(width-64)
                 -2*(width-66)-2*(width-66)-2*(width-68)-2*(width-68)-2*(width-70)-2*(width-70)
                 -2*(width-72)-2*(width-72)-2*(width-74)-2*(width-74)-2*(width-76)-2*(width-76)
                 -2*(width-78),2*k*nb_classes))
        for i in range(39,width-39): 
            for j in range(39,width-39): 
                c_i=[]
                for l in range(1,k+1):
                    [a_i, a_j]=[i,j]+offset[2*l-2]
                    [b_i, b_j]=[i,j]+offset[2*l-1]
                    
                    c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
                C[(i-39)*(width-78)+j-39,]=c_i

        col_cut = np.split(P, [39, width-39], axis=1)
        row_cut = np.split(col_cut[1], [39, width-39])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=dot(listing,L)
        return C,Y
######## 2-D N-DUDE Bound ########
def make_data_for_Two_NN_DUDE_LB(P,k,L,im_bin,nb_classes,n,offset):
    width=int(math.sqrt(n))
    L_split=np.split(L, 2)
    L0=L_split[0]
    L1=L_split[1]
    if k==1:
        col_cut = np.split(P, [1, width-1], axis=1)
        listing = np.reshape(col_cut[1], (col_cut[1].shape[0]*col_cut[1].shape[1], col_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        # Divide case whether the source is 0 or not #
        for i in range(0, width):
            for j in range(1,width-1):
                if im_bin[i,j]==0:
                    Y_i=dot(col_cut[1][i,j-1,], L0)
                    Y[i*(width-2)+j-1,]=Y_i
                else:
                    Y_i=dot(col_cut[1][i,j-1,], L1)
                    Y[i*(width-2)+j-1,]=Y_i
        return Y
    elif 2<=k<=4:
        col_cut = np.split(P, [1, width-1], axis=1)
        row_cut = np.split(col_cut[1], [1, width-1])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(1, width-1):
            for j in range(1,width-1):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-1,j-1,], L0)
                    Y[(i-1)*(width-2)+j-1,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-1,j-1,], L1)
                    Y[(i-1)*(width-2)+j-1,]=Y_i
        return Y
    elif k==5:
        col_cut = np.split(P, [2, width-2], axis=1)
        row_cut = np.split(col_cut[1], [1, width-1])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(1, width-1):
            for j in range(2,width-2):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-1,j-2,], L0)
                    Y[(i-1)*(width-4)+j-2,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-1,j-2,], L1)
                    Y[(i-1)*(width-4)+j-2,]=Y_i
        return Y
    elif 6<=k<=12:
        col_cut = np.split(P, [2, width-2], axis=1)
        row_cut = np.split(col_cut[1], [2, width-2])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(2, width-2):
            for j in range(2,width-2):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-2,j-2,], L0)
                    Y[(i-2)*(width-4)+j-2,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-2,j-2,], L1)
                    Y[(i-2)*(width-4)+j-2,]=Y_i
        return Y
    elif k==13:
        col_cut = np.split(P, [3, width-3], axis=1)
        row_cut = np.split(col_cut[1], [2, width-2])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(2, width-2):
            for j in range(3,width-3):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-2,j-3,], L0)
                    Y[(i-2)*(width-6)+j-3,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-2,j-3,], L1)
                    Y[(i-2)*(width-6)+j-3,]=Y_i
        return Y
    elif 14<=k<=24:
        col_cut = np.split(P, [3, width-3], axis=1)
        row_cut = np.split(col_cut[1], [3, width-3])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(3, width-3):
            for j in range(3,width-3):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-3,j-3,], L0)
                    Y[(i-3)*(width-6)+j-3,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-3,j-3,], L1)
                    Y[(i-3)*(width-6)+j-3,]=Y_i
        return Y
    elif k==25:
        col_cut = np.split(P, [4, width-4], axis=1)
        row_cut = np.split(col_cut[1], [3, width-3])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(3, width-3):
            for j in range(4,width-4):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-3,j-4,], L0)
                    Y[(i-3)*(width-8)+j-4,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-3,j-4,], L1)
                    Y[(i-3)*(width-8)+j-4,]=Y_i
        return Y
    elif 26<=k<=40:
        col_cut = np.split(P, [4, width-4], axis=1)
        row_cut = np.split(col_cut[1], [4, width-4])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(4, width-4):
            for j in range(4,width-4):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-4,j-4,], L0)
                    Y[(i-4)*(width-8)+j-4,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-4,j-4,], L1)
                    Y[(i-4)*(width-8)+j-4,]=Y_i
        return Y
    elif k==41:
        col_cut = np.split(P, [5, width-5], axis=1)
        row_cut = np.split(col_cut[1], [4, width-4])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(4, width-4):
            for j in range(5,width-5):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-4,j-5,], L0)
                    Y[(i-4)*(width-10)+j-5,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-4,j-5,], L1)
                    Y[(i-4)*(width-10)+j-5,]=Y_i
        return Y
    elif 42<=k<=60:
        col_cut = np.split(P, [5, width-5], axis=1)
        row_cut = np.split(col_cut[1], [5, width-5])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(5, width-5):
            for j in range(5,width-5):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-5,j-5,], L0)
                    Y[(i-5)*(width-10)+j-5,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-5,j-5,], L1)
                    Y[(i-5)*(width-10)+j-5,]=Y_i
        return Y
    elif k==61:
        col_cut = np.split(P, [6, width-6], axis=1)
        row_cut = np.split(col_cut[1], [5, width-5])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(5, width-5):
            for j in range(6,width-6):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-5,j-6,], L0)
                    Y[(i-5)*(width-12)+j-6,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-5,j-6,], L1)
                    Y[(i-5)*(width-12)+j-6,]=Y_i
        return Y
    elif 62<=k<=84:
        #C=zeros((n-2*width-2*(width-2)-2*(width-2)-2*(width-4)-2*(width-4)-2*(width-6)-2*(width-6)-2*(width-8)-2*(width-8)
        #         -2*(width-10)-2*(width-10)-2*(width-12),2*k*nb_classes))
        #for i in range(6,width-6): 
        #    for j in range(6,width-6): 
        #        c_i=[]
        #        for l in range(1,k+1):
        #            [a_i, a_j]=[i,j]+offset[2*l-2]
        #            [b_i, b_j]=[i,j]+offset[2*l-1]
        #            
        #            c_i=c_i+P[a_i,a_j,].tolist()+P[b_i,b_j,].tolist()
        #        C[(i-6)*(width-12)+j-6,]=c_i

        col_cut = np.split(P, [6, width-6], axis=1)
        row_cut = np.split(col_cut[1], [6, width-6])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(6, width-6):
            for j in range(6,width-6):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-6,j-6,], L0)
                    Y[(i-6)*(width-12)+j-6,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-6,j-6,], L1)
                    Y[(i-6)*(width-12)+j-6,]=Y_i
        return Y
    elif k==85:
        col_cut = np.split(P, [7, width-7], axis=1)
        row_cut = np.split(col_cut[1], [6, width-6])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(6, width-6):
            for j in range(7,width-7):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-6,j-7,], L0)
                    Y[(i-6)*(width-14)+j-7,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-6,j-7,], L1)
                    Y[(i-6)*(width-14)+j-7,]=Y_i
        return Y
    elif 86<=k<=112:
        col_cut = np.split(P, [7, width-7], axis=1)
        row_cut = np.split(col_cut[1], [7, width-7])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(7, width-7):
            for j in range(7,width-7):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-7,j-7,], L0)
                    Y[(i-7)*(width-14)+j-7,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-7,j-7,], L1)
                    Y[(i-7)*(width-14)+j-7,]=Y_i
        return Y
    elif k==113:
        col_cut = np.split(P, [8, width-8], axis=1)
        row_cut = np.split(col_cut[1], [7, width-7])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(7, width-7):
            for j in range(8,width-8):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-7,j-8,], L0)
                    Y[(i-7)*(width-16)+j-8,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-7,j-8,], L1)
                    Y[(i-7)*(width-16)+j-8,]=Y_i
        return Y
    elif 114<=k<=144:
        col_cut = np.split(P, [8, width-8], axis=1)
        row_cut = np.split(col_cut[1], [8, width-8])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(8, width-8):
            for j in range(8,width-8):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-8,j-8,], L0)
                    Y[(i-8)*(width-16)+j-8,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-8,j-8,], L1)
                    Y[(i-8)*(width-16)+j-8,]=Y_i
        return Y
    elif k==145:
        col_cut = np.split(P, [9, width-9], axis=1)
        row_cut = np.split(col_cut[1], [8, width-8])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(8, width-8):
            for j in range(9,width-9):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-8,j-9,], L0)
                    Y[(i-8)*(width-18)+j-9,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-8,j-9,], L1)
                    Y[(i-8)*(width-18)+j-9,]=Y_i
        return Y
    elif 146<=k<=180:
        col_cut = np.split(P, [9, width-9], axis=1)
        row_cut = np.split(col_cut[1], [9, width-9])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(9, width-9):
            for j in range(9,width-9):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-9,j-9,], L0)
                    Y[(i-9)*(width-18)+j-9,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-9,j-9,], L1)
                    Y[(i-9)*(width-18)+j-9,]=Y_i
        return Y
    elif k==181:
        col_cut = np.split(P, [10, width-10], axis=1)
        row_cut = np.split(col_cut[1], [9, width-9])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(9, width-9):
            for j in range(10,width-10):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-9,j-10,], L0)
                    Y[(i-9)*(width-20)+j-10,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-9,j-10,], L1)
                    Y[(i-9)*(width-20)+j-10,]=Y_i
        return Y
    elif 182<=k<=220:
        col_cut = np.split(P, [10, width-10], axis=1)
        row_cut = np.split(col_cut[1], [10, width-10])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(10, width-10):
            for j in range(10,width-10):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-10,j-10,], L0)
                    Y[(i-10)*(width-20)+j-10,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-10,j-10,], L1)
                    Y[(i-10)*(width-20)+j-10,]=Y_i
        return Y
    elif k==221:
        col_cut = np.split(P, [11, width-11], axis=1)
        row_cut = np.split(col_cut[1], [10, width-10])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(10, width-10):
            for j in range(11,width-11):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-10,j-11,], L0)
                    Y[(i-10)*(width-22)+j-11,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-10,j-11,], L1)
                    Y[(i-10)*(width-22)+j-11,]=Y_i
        return Y
    elif 222<=k<=264:
        col_cut = np.split(P, [11, width-11], axis=1)
        row_cut = np.split(col_cut[1], [11, width-11])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(11, width-11):
            for j in range(11,width-11):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-11,j-11,], L0)
                    Y[(i-11)*(width-22)+j-11,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-11,j-11,], L1)
                    Y[(i-11)*(width-22)+j-11,]=Y_i
        return Y
    elif k==265:
        col_cut = np.split(P, [12, width-12], axis=1)
        row_cut = np.split(col_cut[1], [11, width-11])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(11, width-11):
            for j in range(12,width-12):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-11,j-12,], L0)
                    Y[(i-11)*(width-24)+j-12,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-11,j-12,], L1)
                    Y[(i-11)*(width-24)+j-12,]=Y_i
        return Y
    elif 266<=k<=312:
        col_cut = np.split(P, [12, width-12], axis=1)
        row_cut = np.split(col_cut[1], [12, width-12])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(12, width-12):
            for j in range(12,width-12):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-12,j-12,], L0)
                    Y[(i-12)*(width-24)+j-12,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-12,j-12,], L1)
                    Y[(i-12)*(width-24)+j-12,]=Y_i
        return Y
    elif k==313:
        col_cut = np.split(P, [13, width-13], axis=1)
        row_cut = np.split(col_cut[1], [12, width-12])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(12, width-12):
            for j in range(13,width-13):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-12,j-13,], L0)
                    Y[(i-12)*(width-26)+j-13,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-12,j-13,], L1)
                    Y[(i-12)*(width-26)+j-13,]=Y_i
        return Y
    elif 314<=k<=364:
        col_cut = np.split(P, [13, width-13], axis=1)
        row_cut = np.split(col_cut[1], [13, width-13])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(13, width-13):
            for j in range(13,width-13):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-13,j-13,], L0)
                    Y[(i-13)*(width-26)+j-13,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-13,j-13,], L1)
                    Y[(i-13)*(width-26)+j-13,]=Y_i
        return Y
    elif k==365:
        col_cut = np.split(P, [14, width-14], axis=1)
        row_cut = np.split(col_cut[1], [13, width-13])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(13, width-13):
            for j in range(14,width-14):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-13,j-14,], L0)
                    Y[(i-13)*(width-28)+j-14,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-13,j-14,], L1)
                    Y[(i-13)*(width-28)+j-14,]=Y_i
        return Y
    elif 366<=k<=420:
        col_cut = np.split(P, [14, width-14], axis=1)
        row_cut = np.split(col_cut[1], [14, width-14])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(14, width-14):
            for j in range(14,width-14):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-14,j-14,], L0)
                    Y[(i-14)*(width-28)+j-14,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-14,j-14,], L1)
                    Y[(i-14)*(width-28)+j-14,]=Y_i
        return Y
    elif k==421:
        col_cut = np.split(P, [15, width-15], axis=1)
        row_cut = np.split(col_cut[1], [14, width-14])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(14, width-14):
            for j in range(15,width-15):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-14,j-15,], L0)
                    Y[(i-14)*(width-30)+j-15,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-14,j-15,], L1)
                    Y[(i-14)*(width-30)+j-15,]=Y_i
        return Y
    elif 422<=k<=480:
        col_cut = np.split(P, [15, width-15], axis=1)
        row_cut = np.split(col_cut[1], [15, width-15])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(15, width-15):
            for j in range(15,width-15):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-15,j-15,], L0)
                    Y[(i-15)*(width-30)+j-15,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-15,j-15,], L1)
                    Y[(i-15)*(width-30)+j-15,]=Y_i
        return Y
    elif k==481:
        col_cut = np.split(P, [16, width-16], axis=1)
        row_cut = np.split(col_cut[1], [15, width-15])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(15, width-15):
            for j in range(16,width-16):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-15,j-16,], L0)
                    Y[(i-15)*(width-32)+j-16,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-15,j-16,], L1)
                    Y[(i-15)*(width-32)+j-16,]=Y_i
        return Y
    elif 482<=k<=544:
        col_cut = np.split(P, [16, width-16], axis=1)
        row_cut = np.split(col_cut[1], [16, width-16])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(16, width-16):
            for j in range(16,width-16):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-16,j-16,], L0)
                    Y[(i-16)*(width-32)+j-16,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-16,j-16,], L1)
                    Y[(i-16)*(width-32)+j-16,]=Y_i
        return Y
    elif k==545:
        col_cut = np.split(P, [17, width-17], axis=1)
        row_cut = np.split(col_cut[1], [16, width-16])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(16, width-16):
            for j in range(17,width-17):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-16,j-17,], L0)
                    Y[(i-16)*(width-34)+j-17,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-16,j-17,], L1)
                    Y[(i-16)*(width-34)+j-17,]=Y_i
        return Y
    elif 546<=k<=612:
        col_cut = np.split(P, [17, width-17], axis=1)
        row_cut = np.split(col_cut[1], [17, width-17])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(17, width-17):
            for j in range(17,width-17):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-17,j-17,], L0)
                    Y[(i-17)*(width-34)+j-17,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-17,j-17,], L1)
                    Y[(i-17)*(width-34)+j-17,]=Y_i
        return Y
    elif k==613:
        col_cut = np.split(P, [18, width-18], axis=1)
        row_cut = np.split(col_cut[1], [17, width-17])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(17, width-17):
            for j in range(18,width-18):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-17,j-18,], L0)
                    Y[(i-17)*(width-36)+j-18,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-17,j-18,], L1)
                    Y[(i-17)*(width-36)+j-18,]=Y_i
        return Y
    elif 614<=k<=684:
        col_cut = np.split(P, [18, width-18], axis=1)
        row_cut = np.split(col_cut[1], [18, width-18])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(18, width-18):
            for j in range(18,width-18):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-18,j-18,], L0)
                    Y[(i-18)*(width-36)+j-18,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-18,j-18,], L1)
                    Y[(i-18)*(width-36)+j-18,]=Y_i
        return Y
    elif k==685:
        col_cut = np.split(P, [19, width-19], axis=1)
        row_cut = np.split(col_cut[1], [18, width-18])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(18, width-18):
            for j in range(19,width-19):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-18,j-19,], L0)
                    Y[(i-18)*(width-38)+j-19,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-18,j-19,], L1)
                    Y[(i-18)*(width-38)+j-19,]=Y_i
        return Y
    elif 686<=k<=760:
        col_cut = np.split(P, [19, width-19], axis=1)
        row_cut = np.split(col_cut[1], [19, width-19])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(19, width-19):
            for j in range(19,width-19):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-19,j-19,], L0)
                    Y[(i-19)*(width-38)+j-19,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-19,j-19,], L1)
                    Y[(i-19)*(width-38)+j-19,]=Y_i
        return Y
    elif k==761:
        col_cut = np.split(P, [20, width-20], axis=1)
        row_cut = np.split(col_cut[1], [19, width-19])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(19, width-19):
            for j in range(20,width-20):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-19,j-20,], L0)
                    Y[(i-19)*(width-40)+j-20,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-19,j-20,], L1)
                    Y[(i-19)*(width-40)+j-20,]=Y_i
        return Y
    elif 762<=k<=840:
        col_cut = np.split(P, [20, width-20], axis=1)
        row_cut = np.split(col_cut[1], [20, width-20])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(20, width-20):
            for j in range(20,width-20):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-20,j-20,], L0)
                    Y[(i-20)*(width-40)+j-20,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-20,j-20,], L1)
                    Y[(i-20)*(width-40)+j-20,]=Y_i
        return Y
    elif k==841:
        col_cut = np.split(P, [21, width-21], axis=1)
        row_cut = np.split(col_cut[1], [20, width-20])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(20, width-20):
            for j in range(21,width-21):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-20,j-21,], L0)
                    Y[(i-20)*(width-42)+j-21,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-20,j-21,], L1)
                    Y[(i-20)*(width-42)+j-21,]=Y_i
        return Y
    elif 842<=k<=924:
        col_cut = np.split(P, [21, width-21], axis=1)
        row_cut = np.split(col_cut[1], [21, width-21])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(21, width-21):
            for j in range(21,width-21):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-21,j-21,], L0)
                    Y[(i-21)*(width-42)+j-21,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-21,j-21,], L1)
                    Y[(i-21)*(width-42)+j-21,]=Y_i
        return Y
    elif k==925:
        col_cut = np.split(P, [22, width-22], axis=1)
        row_cut = np.split(col_cut[1], [21, width-21])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(21, width-21):
            for j in range(22,width-22):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-21,j-22,], L0)
                    Y[(i-21)*(width-44)+j-22,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-21,j-22,], L1)
                    Y[(i-21)*(width-44)+j-22,]=Y_i
        return Y
    elif 926<=k<=1012:
        col_cut = np.split(P, [22, width-22], axis=1)
        row_cut = np.split(col_cut[1], [22, width-22])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(22, width-22):
            for j in range(22,width-22):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-22,j-22,], L0)
                    Y[(i-22)*(width-44)+j-22,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-22,j-22,], L1)
                    Y[(i-22)*(width-44)+j-22,]=Y_i
        return Y
    elif k==1013:
        col_cut = np.split(P, [23, width-23], axis=1)
        row_cut = np.split(col_cut[1], [22, width-22])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(22, width-22):
            for j in range(23,width-23):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-22,j-23,], L0)
                    Y[(i-22)*(width-46)+j-23,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-22,j-23,], L1)
                    Y[(i-22)*(width-46)+j-23,]=Y_i
        return Y
    elif 1014<=k<=1104:
        col_cut = np.split(P, [23, width-23], axis=1)
        row_cut = np.split(col_cut[1], [23, width-23])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(23, width-23):
            for j in range(23,width-23):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-23,j-23,], L0)
                    Y[(i-23)*(width-46)+j-23,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-23,j-23,], L1)
                    Y[(i-23)*(width-46)+j-23,]=Y_i
        return Y
    elif k==1105:
        col_cut = np.split(P, [24, width-24], axis=1)
        row_cut = np.split(col_cut[1], [23, width-23])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(23, width-23):
            for j in range(24,width-24):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-23,j-24,], L0)
                    Y[(i-23)*(width-48)+j-24,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-23,j-24,], L1)
                    Y[(i-23)*(width-48)+j-24,]=Y_i
        return Y
    elif 1106<=k<=1200:
        col_cut = np.split(P, [24, width-24], axis=1)
        row_cut = np.split(col_cut[1], [24, width-24])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(24, width-24):
            for j in range(24,width-24):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-24,j-24,], L0)
                    Y[(i-24)*(width-48)+j-24,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-24,j-24,], L1)
                    Y[(i-24)*(width-48)+j-24,]=Y_i
        return Y
    elif k==1201:
        col_cut = np.split(P, [25, width-25], axis=1)
        row_cut = np.split(col_cut[1], [24, width-24])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(24, width-24):
            for j in range(25,width-25):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-24,j-25,], L0)
                    Y[(i-24)*(width-50)+j-25,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-24,j-25,], L1)
                    Y[(i-24)*(width-50)+j-25,]=Y_i
        return Y
    elif 1202<=k<=1300:
        col_cut = np.split(P, [25, width-25], axis=1)
        row_cut = np.split(col_cut[1], [25, width-25])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(25, width-25):
            for j in range(25,width-25):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-25,j-25,], L0)
                    Y[(i-25)*(width-50)+j-25,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-25,j-25,], L1)
                    Y[(i-25)*(width-50)+j-25,]=Y_i
        return Y
    elif k==1301:
        col_cut = np.split(P, [26, width-26], axis=1)
        row_cut = np.split(col_cut[1], [25, width-25])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(25, width-25):
            for j in range(26,width-26):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-25,j-26,], L0)
                    Y[(i-25)*(width-52)+j-26,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-25,j-26,], L1)
                    Y[(i-25)*(width-52)+j-26,]=Y_i
        return Y
    elif 1302<=k<=1404:
        col_cut = np.split(P, [26, width-26], axis=1)
        row_cut = np.split(col_cut[1], [26, width-26])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(26, width-26):
            for j in range(26,width-26):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-26,j-26,], L0)
                    Y[(i-26)*(width-52)+j-26,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-26,j-26,], L1)
                    Y[(i-26)*(width-52)+j-26,]=Y_i
        return Y
    elif k==1405:
        col_cut = np.split(P, [27, width-27], axis=1)
        row_cut = np.split(col_cut[1], [26, width-26])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(26, width-26):
            for j in range(27,width-27):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-26,j-27,], L0)
                    Y[(i-26)*(width-54)+j-27,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-26,j-27,], L1)
                    Y[(i-26)*(width-54)+j-27,]=Y_i
        return Y
    elif 1406<=k<=1512:
        col_cut = np.split(P, [27, width-27], axis=1)
        row_cut = np.split(col_cut[1], [27, width-27])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(27, width-27):
            for j in range(27,width-27):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-27,j-27,], L0)
                    Y[(i-27)*(width-54)+j-27,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-27,j-27,], L1)
                    Y[(i-27)*(width-54)+j-27,]=Y_i
        return Y
    elif k==1513:
        col_cut = np.split(P, [28, width-28], axis=1)
        row_cut = np.split(col_cut[1], [27, width-27])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(27, width-27):
            for j in range(28,width-28):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-27,j-28,], L0)
                    Y[(i-27)*(width-56)+j-28,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-27,j-28,], L1)
                    Y[(i-27)*(width-56)+j-28,]=Y_i
        return Y
    elif 1514<=k<=1624:
        col_cut = np.split(P, [28, width-28], axis=1)
        row_cut = np.split(col_cut[1], [28, width-28])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(28, width-28):
            for j in range(28,width-28):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-28,j-28,], L0)
                    Y[(i-28)*(width-56)+j-28,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-28,j-28,], L1)
                    Y[(i-28)*(width-56)+j-28,]=Y_i
        return Y
    elif k==1625:
        col_cut = np.split(P, [29, width-29], axis=1)
        row_cut = np.split(col_cut[1], [28, width-28])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(28, width-28):
            for j in range(29,width-29):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-28,j-29,], L0)
                    Y[(i-28)*(width-58)+j-29,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-28,j-29,], L1)
                    Y[(i-28)*(width-58)+j-29,]=Y_i
        return Y
    elif 1626<=k<=1740:
        col_cut = np.split(P, [29, width-29], axis=1)
        row_cut = np.split(col_cut[1], [29, width-29])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(29, width-29):
            for j in range(29,width-29):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-29,j-29,], L0)
                    Y[(i-29)*(width-58)+j-29,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-29,j-29,], L1)
                    Y[(i-29)*(width-58)+j-29,]=Y_i
        return Y
    elif k==1741:
        col_cut = np.split(P, [30, width-30], axis=1)
        row_cut = np.split(col_cut[1], [29, width-29])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(29, width-29):
            for j in range(30,width-30):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-29,j-30,], L0)
                    Y[(i-29)*(width-60)+j-30,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-29,j-30,], L1)
                    Y[(i-29)*(width-60)+j-30,]=Y_i
        return Y
    elif 1742<=k<=1860:
        col_cut = np.split(P, [30, width-30], axis=1)
        row_cut = np.split(col_cut[1], [30, width-30])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(30, width-30):
            for j in range(30,width-30):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-30,j-30,], L0)
                    Y[(i-30)*(width-60)+j-30,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-30,j-30,], L1)
                    Y[(i-30)*(width-60)+j-30,]=Y_i
        return Y
    elif k==1861:
        col_cut = np.split(P, [31, width-31], axis=1)
        row_cut = np.split(col_cut[1], [30, width-30])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(30, width-30):
            for j in range(31,width-31):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-30,j-31,], L0)
                    Y[(i-30)*(width-62)+j-31,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-30,j-31,], L1)
                    Y[(i-30)*(width-62)+j-31,]=Y_i
        return Y
    elif 1862<=k<=1984:
        col_cut = np.split(P, [31, width-31], axis=1)
        row_cut = np.split(col_cut[1], [31, width-31])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(31, width-31):
            for j in range(31,width-31):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-31,j-31,], L0)
                    Y[(i-31)*(width-62)+j-31,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-31,j-31,], L1)
                    Y[(i-31)*(width-62)+j-31,]=Y_i
        return Y
    elif k==1985:
        col_cut = np.split(P, [32, width-32], axis=1)
        row_cut = np.split(col_cut[1], [31, width-31])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(31, width-31):
            for j in range(32,width-32):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-31,j-32,], L0)
                    Y[(i-31)*(width-64)+j-32,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-31,j-32,], L1)
                    Y[(i-31)*(width-64)+j-32,]=Y_i
        return Y
    elif 1986<=k<=2112:
        col_cut = np.split(P, [32, width-32], axis=1)
        row_cut = np.split(col_cut[1], [32, width-32])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(32, width-32):
            for j in range(32,width-32):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-32,j-32,], L0)
                    Y[(i-32)*(width-64)+j-32,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-32,j-32,], L1)
                    Y[(i-32)*(width-64)+j-32,]=Y_i
        return Y
    elif k==2113:
        col_cut = np.split(P, [33, width-33], axis=1)
        row_cut = np.split(col_cut[1], [32, width-32])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(32, width-32):
            for j in range(33,width-33):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-32,j-33,], L0)
                    Y[(i-32)*(width-66)+j-33,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-32,j-33,], L1)
                    Y[(i-32)*(width-66)+j-33,]=Y_i
        return Y
    elif 2114<=k<=2244:
        col_cut = np.split(P, [33, width-33], axis=1)
        row_cut = np.split(col_cut[1], [33, width-33])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(33, width-33):
            for j in range(33,width-33):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-33,j-33,], L0)
                    Y[(i-33)*(width-66)+j-33,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-33,j-33,], L1)
                    Y[(i-33)*(width-66)+j-33,]=Y_i
        return Y
    elif k==2245:
        col_cut = np.split(P, [34, width-34], axis=1)
        row_cut = np.split(col_cut[1], [33, width-33])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(33, width-33):
            for j in range(34,width-34):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-33,j-34,], L0)
                    Y[(i-33)*(width-68)+j-34,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-33,j-34,], L1)
                    Y[(i-33)*(width-68)+j-34,]=Y_i
        return Y
    elif 2246<=k<=2380:
        col_cut = np.split(P, [34, width-34], axis=1)
        row_cut = np.split(col_cut[1], [34, width-34])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(34, width-34):
            for j in range(34,width-34):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-34,j-34,], L0)
                    Y[(i-34)*(width-68)+j-34,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-34,j-34,], L1)
                    Y[(i-34)*(width-68)+j-34,]=Y_i
        return Y
    elif k==2381:
        col_cut = np.split(P, [35, width-35], axis=1)
        row_cut = np.split(col_cut[1], [34, width-34])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(34, width-34):
            for j in range(35,width-35):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-34,j-35,], L0)
                    Y[(i-34)*(width-70)+j-35,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-34,j-35,], L1)
                    Y[(i-34)*(width-70)+j-35,]=Y_i
        return Y
    elif 2382<=k<=2520:
        col_cut = np.split(P, [35, width-35], axis=1)
        row_cut = np.split(col_cut[1], [35, width-35])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(35, width-35):
            for j in range(35,width-35):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-35,j-35,], L0)
                    Y[(i-35)*(width-70)+j-35,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-35,j-35,], L1)
                    Y[(i-35)*(width-70)+j-35,]=Y_i
        return Y
    elif k==2521:
        col_cut = np.split(P, [36, width-36], axis=1)
        row_cut = np.split(col_cut[1], [35, width-35])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(35, width-35):
            for j in range(36,width-36):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-35,j-36,], L0)
                    Y[(i-35)*(width-72)+j-36,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-35,j-36,], L1)
                    Y[(i-35)*(width-72)+j-36,]=Y_i
        return Y
    elif 2522<=k<=2664:
        col_cut = np.split(P, [36, width-36], axis=1)
        row_cut = np.split(col_cut[1], [36, width-36])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(36, width-36):
            for j in range(36,width-36):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-36,j-36,], L0)
                    Y[(i-36)*(width-72)+j-36,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-36,j-36,], L1)
                    Y[(i-36)*(width-72)+j-36,]=Y_i
        return Y
    elif k==2665:
        col_cut = np.split(P, [37, width-37], axis=1)
        row_cut = np.split(col_cut[1], [36, width-36])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(36, width-36):
            for j in range(37,width-37):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-36,j-37,], L0)
                    Y[(i-36)*(width-74)+j-37,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-36,j-37,], L1)
                    Y[(i-36)*(width-74)+j-37,]=Y_i
        return Y
    elif 2666<=k<=2812:
        col_cut = np.split(P, [37, width-37], axis=1)
        row_cut = np.split(col_cut[1], [37, width-37])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(37, width-37):
            for j in range(37,width-37):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-37,j-37,], L0)
                    Y[(i-37)*(width-74)+j-37,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-37,j-37,], L1)
                    Y[(i-37)*(width-74)+j-37,]=Y_i
        return Y
    elif k==2813:
        col_cut = np.split(P, [38, width-38], axis=1)
        row_cut = np.split(col_cut[1], [37, width-37])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(37, width-37):
            for j in range(38,width-38):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-37,j-38,], L0)
                    Y[(i-37)*(width-76)+j-38,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-37,j-38,], L1)
                    Y[(i-37)*(width-76)+j-38,]=Y_i
        return Y
    elif 2814<=k<=2964:
        col_cut = np.split(P, [38, width-38], axis=1)
        row_cut = np.split(col_cut[1], [38, width-38])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(38, width-38):
            for j in range(38,width-38):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-38,j-38,], L0)
                    Y[(i-38)*(width-76)+j-38,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-38,j-38,], L1)
                    Y[(i-38)*(width-76)+j-38,]=Y_i
        return Y
    elif k==2965:
        col_cut = np.split(P, [39, width-39], axis=1)
        row_cut = np.split(col_cut[1], [38, width-38])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(38, width-38):
            for j in range(39,width-39):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-38,j-39,], L0)
                    Y[(i-38)*(width-78)+j-39,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-38,j-39,], L1)
                    Y[(i-38)*(width-78)+j-39,]=Y_i
        return Y
    elif 2966<=k<=3120:
        col_cut = np.split(P, [39, width-39], axis=1)
        row_cut = np.split(col_cut[1], [39, width-39])
        listing = np.reshape(row_cut[1], (row_cut[1].shape[0]*row_cut[1].shape[1], row_cut[1].shape[2]))
        Y=zeros((listing.shape[0], L.shape[1]))
        for i in range(39, width-39):
            for j in range(39,width-39):
                if im_bin[i,j]==0:
                    Y_i=dot(row_cut[1][i-39,j-39,], L0)
                    Y[(i-39)*(width-78)+j-39,]=Y_i
                else:
                    Y_i=dot(row_cut[1][i-39,j-39,], L1)
                    Y[(i-39)*(width-78)+j-39,]=Y_i
        return Y
######## 2-D N-DUDE & Bound s_hat generation ########
def mapping_mat_resize(pred_class,k,n):
    width = int(math.sqrt(n))
    if k==1:
        resize = pred_class.reshape(width, width-2)
        s_nn_hat = hstack((zeros((width, 1)), resize, zeros((width, 1))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2<=k<=4:
        resize = pred_class.reshape(width-2, width-2)
        s_nn_hat = hstack((zeros((width-2, 1)), resize, zeros((width-2, 1))))
        s_nn_hat = vstack((zeros((1, width)), s_nn_hat, zeros((1, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==5:
        resize = pred_class.reshape(width-2, width-4)
        s_nn_hat = hstack((zeros((width-2, 2)), resize, zeros((width-2, 2))))
        s_nn_hat = vstack((zeros((1, width)), s_nn_hat, zeros((1, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 6<=k<=12:
        resize = pred_class.reshape(width-4, width-4)
        s_nn_hat = hstack((zeros((width-4, 2)), resize, zeros((width-4, 2))))
        s_nn_hat = vstack((zeros((2, width)), s_nn_hat, zeros((2, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==13:
        resize = pred_class.reshape(width-4, width-6)
        s_nn_hat = hstack((zeros((width-4, 3)), resize, zeros((width-4, 3))))
        s_nn_hat = vstack((zeros((2, width)), s_nn_hat, zeros((2, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat                        
    elif 14<=k<=24:
        resize = pred_class.reshape(width-6, width-6)
        s_nn_hat = hstack((zeros((width-6, 3)), resize, zeros((width-6, 3))))
        s_nn_hat = vstack((zeros((3, width)), s_nn_hat, zeros((3, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat                        
    elif k==25:
        resize = pred_class.reshape(width-6, width-8)
        s_nn_hat = hstack((zeros((width-6, 4)), resize, zeros((width-6, 4))))
        s_nn_hat = vstack((zeros((3, width)), s_nn_hat, zeros((3, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat                        
    elif 26<=k<=40:
        resize = pred_class.reshape(width-8, width-8)
        s_nn_hat = hstack((zeros((width-8, 4)), resize, zeros((width-8, 4))))
        s_nn_hat = vstack((zeros((4, width)), s_nn_hat, zeros((4, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==41:
        resize = pred_class.reshape(width-8, width-10)
        s_nn_hat = hstack((zeros((width-8, 5)), resize, zeros((width-8, 5))))
        s_nn_hat = vstack((zeros((4, width)), s_nn_hat, zeros((4, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 42<=k<=60:
        resize = pred_class.reshape(width-10, width-10)
        s_nn_hat = hstack((zeros((width-10, 5)), resize, zeros((width-10, 5))))
        s_nn_hat = vstack((zeros((5, width)), s_nn_hat, zeros((5, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==61:
        resize = pred_class.reshape(width-10, width-12)
        s_nn_hat = hstack((zeros((width-10, 6)), resize, zeros((width-10, 6))))
        s_nn_hat = vstack((zeros((5, width)), s_nn_hat, zeros((5, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 62<=k<=84:
        resize = pred_class.reshape(width-12, width-12)
        s_nn_hat = hstack((zeros((width-12, 6)), resize, zeros((width-12, 6))))
        s_nn_hat = vstack((zeros((6, width)), s_nn_hat, zeros((6, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat 
    elif k==85:
        resize = pred_class.reshape(width-12, width-14)
        s_nn_hat = hstack((zeros((width-12, 7)), resize, zeros((width-12, 7))))
        s_nn_hat = vstack((zeros((6, width)), s_nn_hat, zeros((6, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 86<=k<=112:
        resize = pred_class.reshape(width-14, width-14)
        s_nn_hat = hstack((zeros((width-14, 7)), resize, zeros((width-14, 7))))
        s_nn_hat = vstack((zeros((7, width)), s_nn_hat, zeros((7, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat 
    elif k==113:
        resize = pred_class.reshape(width-14, width-16)
        s_nn_hat = hstack((zeros((width-14, 8)), resize, zeros((width-14, 8))))
        s_nn_hat = vstack((zeros((7, width)), s_nn_hat, zeros((7, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 114<=k<=144:
        resize = pred_class.reshape(width-16, width-16)
        s_nn_hat = hstack((zeros((width-16, 8)), resize, zeros((width-16, 8))))
        s_nn_hat = vstack((zeros((8, width)), s_nn_hat, zeros((8, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==145:
        resize = pred_class.reshape(width-16, width-18)
        s_nn_hat = hstack((zeros((width-16, 9)), resize, zeros((width-16, 9))))
        s_nn_hat = vstack((zeros((8, width)), s_nn_hat, zeros((8, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 146<=k<=180:
        resize = pred_class.reshape(width-18, width-18)
        s_nn_hat = hstack((zeros((width-18, 9)), resize, zeros((width-18, 9))))
        s_nn_hat = vstack((zeros((9, width)), s_nn_hat, zeros((9, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==181:
        resize = pred_class.reshape(width-18, width-20)
        s_nn_hat = hstack((zeros((width-18, 10)), resize, zeros((width-18, 10))))
        s_nn_hat = vstack((zeros((9, width)), s_nn_hat, zeros((9, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 182<=k<=220:
        resize = pred_class.reshape(width-20, width-20)
        s_nn_hat = hstack((zeros((width-20, 10)), resize, zeros((width-20, 10))))
        s_nn_hat = vstack((zeros((10, width)), s_nn_hat, zeros((10, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==221:
        resize = pred_class.reshape(width-20, width-22)
        s_nn_hat = hstack((zeros((width-20, 11)), resize, zeros((width-20, 11))))
        s_nn_hat = vstack((zeros((10, width)), s_nn_hat, zeros((10, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 222<=k<=264:
        resize = pred_class.reshape(width-22, width-22)
        s_nn_hat = hstack((zeros((width-22, 11)), resize, zeros((width-22, 11))))
        s_nn_hat = vstack((zeros((11, width)), s_nn_hat, zeros((11, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==265:
        resize = pred_class.reshape(width-22, width-24)
        s_nn_hat = hstack((zeros((width-22, 12)), resize, zeros((width-22, 12))))
        s_nn_hat = vstack((zeros((11, width)), s_nn_hat, zeros((11, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 266<=k<=312:
        resize = pred_class.reshape(width-24, width-24)
        s_nn_hat = hstack((zeros((width-24, 12)), resize, zeros((width-24, 12))))
        s_nn_hat = vstack((zeros((12, width)), s_nn_hat, zeros((12, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==313:
        resize = pred_class.reshape(width-24, width-26)
        s_nn_hat = hstack((zeros((width-24, 13)), resize, zeros((width-24, 13))))
        s_nn_hat = vstack((zeros((12, width)), s_nn_hat, zeros((12, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 314<=k<=364:
        resize = pred_class.reshape(width-26, width-26)
        s_nn_hat = hstack((zeros((width-26, 13)), resize, zeros((width-26, 13))))
        s_nn_hat = vstack((zeros((13, width)), s_nn_hat, zeros((13, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==365:
        resize = pred_class.reshape(width-26, width-28)
        s_nn_hat = hstack((zeros((width-26, 14)), resize, zeros((width-26, 14))))
        s_nn_hat = vstack((zeros((13, width)), s_nn_hat, zeros((13, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 366<=k<=420:
        resize = pred_class.reshape(width-28, width-28)
        s_nn_hat = hstack((zeros((width-28, 14)), resize, zeros((width-28, 14))))
        s_nn_hat = vstack((zeros((14, width)), s_nn_hat, zeros((14, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==421:
        resize = pred_class.reshape(width-28, width-30)
        s_nn_hat = hstack((zeros((width-28, 15)), resize, zeros((width-28, 15))))
        s_nn_hat = vstack((zeros((14, width)), s_nn_hat, zeros((14, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 422<=k<=480:
        resize = pred_class.reshape(width-30, width-30)
        s_nn_hat = hstack((zeros((width-30, 15)), resize, zeros((width-30, 15))))
        s_nn_hat = vstack((zeros((15, width)), s_nn_hat, zeros((15, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==481:
        resize = pred_class.reshape(width-30, width-32)
        s_nn_hat = hstack((zeros((width-30, 16)), resize, zeros((width-30, 16))))
        s_nn_hat = vstack((zeros((15, width)), s_nn_hat, zeros((15, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 482<=k<=544:
        resize = pred_class.reshape(width-32, width-32)
        s_nn_hat = hstack((zeros((width-32, 16)), resize, zeros((width-32, 16))))
        s_nn_hat = vstack((zeros((16, width)), s_nn_hat, zeros((16, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==545:
        resize = pred_class.reshape(width-32, width-34)
        s_nn_hat = hstack((zeros((width-32, 17)), resize, zeros((width-32, 17))))
        s_nn_hat = vstack((zeros((16, width)), s_nn_hat, zeros((16, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 546<=k<=612:
        resize = pred_class.reshape(width-34, width-34)
        s_nn_hat = hstack((zeros((width-34, 17)), resize, zeros((width-34, 17))))
        s_nn_hat = vstack((zeros((17, width)), s_nn_hat, zeros((17, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==613:
        resize = pred_class.reshape(width-34, width-36)
        s_nn_hat = hstack((zeros((width-34, 18)), resize, zeros((width-34, 18))))
        s_nn_hat = vstack((zeros((17, width)), s_nn_hat, zeros((17, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 614<=k<=684:
        resize = pred_class.reshape(width-36, width-36)
        s_nn_hat = hstack((zeros((width-36, 18)), resize, zeros((width-36, 18))))
        s_nn_hat = vstack((zeros((18, width)), s_nn_hat, zeros((18, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==685:
        resize = pred_class.reshape(width-36, width-38)
        s_nn_hat = hstack((zeros((width-36, 19)), resize, zeros((width-36, 19))))
        s_nn_hat = vstack((zeros((18, width)), s_nn_hat, zeros((18, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 686<=k<=760:
        resize = pred_class.reshape(width-38, width-38)
        s_nn_hat = hstack((zeros((width-38, 19)), resize, zeros((width-38, 19))))
        s_nn_hat = vstack((zeros((19, width)), s_nn_hat, zeros((19, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==761:
        resize = pred_class.reshape(width-38, width-40)
        s_nn_hat = hstack((zeros((width-38, 20)), resize, zeros((width-38, 20))))
        s_nn_hat = vstack((zeros((19, width)), s_nn_hat, zeros((19, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 762<=k<=840:
        resize = pred_class.reshape(width-40, width-40)
        s_nn_hat = hstack((zeros((width-40, 20)), resize, zeros((width-40, 20))))
        s_nn_hat = vstack((zeros((20, width)), s_nn_hat, zeros((20, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==841:
        resize = pred_class.reshape(width-40, width-42)
        s_nn_hat = hstack((zeros((width-40, 21)), resize, zeros((width-40, 21))))
        s_nn_hat = vstack((zeros((20, width)), s_nn_hat, zeros((20, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 842<=k<=924:
        resize = pred_class.reshape(width-42, width-42)
        s_nn_hat = hstack((zeros((width-42, 21)), resize, zeros((width-42, 21))))
        s_nn_hat = vstack((zeros((21, width)), s_nn_hat, zeros((21, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==925:
        resize = pred_class.reshape(width-42, width-44)
        s_nn_hat = hstack((zeros((width-42, 22)), resize, zeros((width-42, 22))))
        s_nn_hat = vstack((zeros((21, width)), s_nn_hat, zeros((21, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 926<=k<=1012:
        resize = pred_class.reshape(width-44, width-44)
        s_nn_hat = hstack((zeros((width-44, 22)), resize, zeros((width-44, 22))))
        s_nn_hat = vstack((zeros((22, width)), s_nn_hat, zeros((22, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1013:
        resize = pred_class.reshape(width-44, width-46)
        s_nn_hat = hstack((zeros((width-44, 23)), resize, zeros((width-44, 23))))
        s_nn_hat = vstack((zeros((22, width)), s_nn_hat, zeros((22, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1014<=k<=1104:
        resize = pred_class.reshape(width-46, width-46)
        s_nn_hat = hstack((zeros((width-46, 23)), resize, zeros((width-46, 23))))
        s_nn_hat = vstack((zeros((23, width)), s_nn_hat, zeros((23, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1105:
        resize = pred_class.reshape(width-46, width-48)
        s_nn_hat = hstack((zeros((width-46, 24)), resize, zeros((width-46, 24))))
        s_nn_hat = vstack((zeros((23, width)), s_nn_hat, zeros((23, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1106<=k<=1200:
        resize = pred_class.reshape(width-48, width-48)
        s_nn_hat = hstack((zeros((width-48, 24)), resize, zeros((width-48, 24))))
        s_nn_hat = vstack((zeros((24, width)), s_nn_hat, zeros((24, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1201:
        resize = pred_class.reshape(width-48, width-50)
        s_nn_hat = hstack((zeros((width-48, 25)), resize, zeros((width-48, 25))))
        s_nn_hat = vstack((zeros((24, width)), s_nn_hat, zeros((24, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1202<=k<=1300:
        resize = pred_class.reshape(width-50, width-50)
        s_nn_hat = hstack((zeros((width-50, 25)), resize, zeros((width-50, 25))))
        s_nn_hat = vstack((zeros((25, width)), s_nn_hat, zeros((25, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1301:
        resize = pred_class.reshape(width-50, width-52)
        s_nn_hat = hstack((zeros((width-50, 26)), resize, zeros((width-50, 26))))
        s_nn_hat = vstack((zeros((25, width)), s_nn_hat, zeros((25, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1302<=k<=1404:
        resize = pred_class.reshape(width-52, width-52)
        s_nn_hat = hstack((zeros((width-52, 26)), resize, zeros((width-52, 26))))
        s_nn_hat = vstack((zeros((26, width)), s_nn_hat, zeros((26, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1405:
        resize = pred_class.reshape(width-52, width-54)
        s_nn_hat = hstack((zeros((width-52, 27)), resize, zeros((width-52, 27))))
        s_nn_hat = vstack((zeros((26, width)), s_nn_hat, zeros((26, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1406<=k<=1512:
        resize = pred_class.reshape(width-54, width-54)
        s_nn_hat = hstack((zeros((width-54, 27)), resize, zeros((width-54, 27))))
        s_nn_hat = vstack((zeros((27, width)), s_nn_hat, zeros((27, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1513:
        resize = pred_class.reshape(width-54, width-56)
        s_nn_hat = hstack((zeros((width-54, 28)), resize, zeros((width-54, 28))))
        s_nn_hat = vstack((zeros((27, width)), s_nn_hat, zeros((27, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1514<=k<=1624:
        resize = pred_class.reshape(width-56, width-56)
        s_nn_hat = hstack((zeros((width-56, 28)), resize, zeros((width-56, 28))))
        s_nn_hat = vstack((zeros((28, width)), s_nn_hat, zeros((28, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1625:
        resize = pred_class.reshape(width-56, width-58)
        s_nn_hat = hstack((zeros((width-56, 29)), resize, zeros((width-56, 29))))
        s_nn_hat = vstack((zeros((28, width)), s_nn_hat, zeros((28, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1626<=k<=1740:
        resize = pred_class.reshape(width-58, width-58)
        s_nn_hat = hstack((zeros((width-58, 29)), resize, zeros((width-58, 29))))
        s_nn_hat = vstack((zeros((29, width)), s_nn_hat, zeros((29, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1741:
        resize = pred_class.reshape(width-58, width-60)
        s_nn_hat = hstack((zeros((width-58, 30)), resize, zeros((width-58, 30))))
        s_nn_hat = vstack((zeros((29, width)), s_nn_hat, zeros((29, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1742<=k<=1860:
        resize = pred_class.reshape(width-60, width-60)
        s_nn_hat = hstack((zeros((width-60, 30)), resize, zeros((width-60, 30))))
        s_nn_hat = vstack((zeros((30, width)), s_nn_hat, zeros((30, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1861:
        resize = pred_class.reshape(width-60, width-62)
        s_nn_hat = hstack((zeros((width-60, 31)), resize, zeros((width-60, 31))))
        s_nn_hat = vstack((zeros((30, width)), s_nn_hat, zeros((30, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1862<=k<=1984:
        resize = pred_class.reshape(width-62, width-62)
        s_nn_hat = hstack((zeros((width-62, 31)), resize, zeros((width-62, 31))))
        s_nn_hat = vstack((zeros((31, width)), s_nn_hat, zeros((31, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==1985:
        resize = pred_class.reshape(width-62, width-64)
        s_nn_hat = hstack((zeros((width-62, 32)), resize, zeros((width-62, 32))))
        s_nn_hat = vstack((zeros((31, width)), s_nn_hat, zeros((31, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 1986<=k<=2112:
        resize = pred_class.reshape(width-64, width-64)
        s_nn_hat = hstack((zeros((width-64, 32)), resize, zeros((width-64, 32))))
        s_nn_hat = vstack((zeros((32, width)), s_nn_hat, zeros((32, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2113:
        resize = pred_class.reshape(width-64, width-66)
        s_nn_hat = hstack((zeros((width-64, 33)), resize, zeros((width-64, 33))))
        s_nn_hat = vstack((zeros((32, width)), s_nn_hat, zeros((32, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2114<=k<=2244:
        resize = pred_class.reshape(width-66, width-66)
        s_nn_hat = hstack((zeros((width-66, 33)), resize, zeros((width-66, 33))))
        s_nn_hat = vstack((zeros((33, width)), s_nn_hat, zeros((33, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2245:
        resize = pred_class.reshape(width-66, width-68)
        s_nn_hat = hstack((zeros((width-66, 34)), resize, zeros((width-66, 34))))
        s_nn_hat = vstack((zeros((33, width)), s_nn_hat, zeros((33, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2246<=k<=2380:
        resize = pred_class.reshape(width-68, width-68)
        s_nn_hat = hstack((zeros((width-68, 34)), resize, zeros((width-68, 34))))
        s_nn_hat = vstack((zeros((34, width)), s_nn_hat, zeros((34, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2381:
        resize = pred_class.reshape(width-68, width-70)
        s_nn_hat = hstack((zeros((width-68, 35)), resize, zeros((width-68, 35))))
        s_nn_hat = vstack((zeros((34, width)), s_nn_hat, zeros((34, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2382<=k<=2520:
        resize = pred_class.reshape(width-70, width-70)
        s_nn_hat = hstack((zeros((width-70, 35)), resize, zeros((width-70, 35))))
        s_nn_hat = vstack((zeros((35, width)), s_nn_hat, zeros((35, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2521:
        resize = pred_class.reshape(width-70, width-72)
        s_nn_hat = hstack((zeros((width-70, 36)), resize, zeros((width-70, 36))))
        s_nn_hat = vstack((zeros((35, width)), s_nn_hat, zeros((35, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2522<=k<=2664:
        resize = pred_class.reshape(width-72, width-72)
        s_nn_hat = hstack((zeros((width-72, 36)), resize, zeros((width-72, 36))))
        s_nn_hat = vstack((zeros((36, width)), s_nn_hat, zeros((36, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2665:
        resize = pred_class.reshape(width-72, width-74)
        s_nn_hat = hstack((zeros((width-72, 37)), resize, zeros((width-72, 37))))
        s_nn_hat = vstack((zeros((36, width)), s_nn_hat, zeros((36, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2666<=k<=2812:
        resize = pred_class.reshape(width-74, width-74)
        s_nn_hat = hstack((zeros((width-74, 37)), resize, zeros((width-74, 37))))
        s_nn_hat = vstack((zeros((37, width)), s_nn_hat, zeros((37, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2813:
        resize = pred_class.reshape(width-74, width-76)
        s_nn_hat = hstack((zeros((width-74, 38)), resize, zeros((width-74, 38))))
        s_nn_hat = vstack((zeros((37, width)), s_nn_hat, zeros((37, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2814<=k<=2964:
        resize = pred_class.reshape(width-76, width-76)
        s_nn_hat = hstack((zeros((width-76, 38)), resize, zeros((width-76, 38))))
        s_nn_hat = vstack((zeros((38, width)), s_nn_hat, zeros((38, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif k==2965:
        resize = pred_class.reshape(width-76, width-78)
        s_nn_hat = hstack((zeros((width-76, 39)), resize, zeros((width-76, 39))))
        s_nn_hat = vstack((zeros((38, width)), s_nn_hat, zeros((38, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
    elif 2966<=k<=3120:
        resize = pred_class.reshape(width-78, width-78)
        s_nn_hat = hstack((zeros((width-78, 39)), resize, zeros((width-78, 39))))
        s_nn_hat = vstack((zeros((39, width)), s_nn_hat, zeros((39, width))))
        s_nn_hat = s_nn_hat.reshape(n,)
        return s_nn_hat
######## 2-D N-DUDE Padding ########
def make_data_for_Two_NN_DUDE_PD(P_padding,P,k,L,nb_classes,n,offset,extended_dp):
    width=int(math.sqrt(extended_dp))
    C=zeros((n, 2*k*nb_classes))
    for i in range(39,width-39): 
        for j in range(39,width-39):
            c_i=[]
            for l in range(1,k+1):
                [a_i, a_j]=[i,j]+offset[2*l-2]
                [b_i, b_j]=[i,j]+offset[2*l-1]
                
                c_i=c_i+P_padding[a_i,a_j,].tolist()+P_padding[b_i,b_j,].tolist()
            C[(i-39)*(width-78)+j-39,]=c_i
    listing = np.reshape(P, (n, nb_classes))
    Y=dot(listing,L)
    return C,Y

######## 2-D N-DUDE Padding Bound ########
def make_data_for_Two_NN_DUDE_PD_LB(P_padding,P,im_bin,k,L_lower,nb_classes,n,offset,extended_dp):
    width=int(math.sqrt(extended_dp))
    L_split=np.split(L_lower, 2)
    L0=L_split[0]
    L1=L_split[1]
    
    listing = np.reshape(P, (n, nb_classes))
    Y=zeros((listing.shape[0], L_lower.shape[1]))
    for i in range(39, width-39):
        for j in range(39,width-39):
            if im_bin[i-39,j-39]==0:
                Y_i=dot(listing[(i-39)*(width-78)+j-39,:], L0)
                Y[(i-39)*(width-78)+j-39,]=Y_i
            else:
                Y_i=dot(listing[(i-39)*(width-78)+j-39,:], L1)
                Y[(i-39)*(width-78)+j-39,]=Y_i
    return Y
   
######## 2-D N-DUDE & Bound & Padding denoising ########
def denoise_with_s_Two_NN_DUDE(z,s): # No need k #
    n=len(z)
    x_hat=z.copy()
    for i in range(0,n): # look all-points. 
        if s[i]==0: # Anyway, We stick zeros in resizing. So losing Datapoint's mapping 0 which do not anything to that point #
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat
