import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN, device=gpu1,floatX=float32"
import theano
import keras
import time
import sys

import numpy as np
import Binary_DUDE as DUDE
import Binary_N_DUDE as N_DUDE

from numpy import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, GRU
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D
from keras.layers import Merge

dim = int(sys.argv[1])
### Pre-Generated Data Load ###
data=np.load('/home/ubuntu/Yoon_ICE/Data_Generation&Save/Neural_dude_Data_lena512.npz')

nb_classes=data['nb_classes']
delta=data['delta']
dp=data['dp']
loss_lines=data['loss_lines']
imarray=data['imarray']
im_bin=data['im_bin']
x=data['x']
z=data['z']
z_two=data['z_two']
L=data['L']
L_lower=data['L_lower']
L_new=data['L_new']
offset=data['offset']

### Make data which this experiment needs ###
n=imarray.shape[0]*imarray.shape[1]
alpha_size=2
mapping_size=3
k_max=40

Z=[]
P=np.zeros((len(delta),imarray.shape[0],imarray.shape[1],nb_classes),dtype=np.int)
P_padding=np.zeros((len(delta), imarray.shape[0]+2*loss_lines, imarray.shape[1]+2*loss_lines, nb_classes), 
                   dtype=np.int)

for i in range(len(delta)):
    Temp=np_utils.to_categorical(z[i],nb_classes) ## [0]->[1 0], [1]->[0 1]
    Temp=Temp.reshape(nb_classes*n,)
    Z=np.append(Z,Temp)
    
Z=np.reshape(Z, (len(delta)*n,nb_classes))

for i in range(len(delta)):
    P[i]=np.reshape(Z[i*n:(i+1)*n,],(imarray.shape[0], imarray.shape[1], nb_classes))

for i in range(len(delta)):
    Temp=hstack((zeros((imarray.shape[0],loss_lines,nb_classes)), P[i], zeros((imarray.shape[0],loss_lines,nb_classes))))
    P_padding[i]=vstack((zeros((loss_lines,Temp.shape[1],nb_classes)), Temp, zeros((loss_lines,Temp.shape[1],nb_classes))))

### True Loss Mat ###
Error_One_NN_DUDE=zeros((len(delta),k_max+1))
Error_One_NN_DUDE_LB=Error_One_NN_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):    
    Error_One_NN_DUDE[i,0]=delta[i]
    Error_One_NN_DUDE_LB[i,0]=delta[i]

### Est Loss Mat ###
Est_Loss_One_NN_DUDE=zeros((len(delta),k_max+1))

for i in range(len(delta)):
    Est_Loss_One_NN_DUDE[i,0]=delta[i] # 1-D N-DUDE
    
### X_hat Mat ###
X_hat_One_NN_DUDE=zeros((len(delta)*k_max,n))
X_hat_One_NN_DUDE_LB=X_hat_One_NN_DUDE.copy()


for i in range(0,1):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        
        ### 1-D N-DUDE ###
        C,Y = N_DUDE.make_data_for_rnn_n_dude(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        C_left = np.split(C,2,axis=1)[0]
        C_right = np.split(C,2,axis=1)[1]
        left_context = Sequential()
        left_context.add(LSTM(dim,init='he_normal',return_sequences=False, input_shape=C_left.shape[1:]))
        right_context = Sequential()
        right_context.add(LSTM(dim,init='he_normal',return_sequences=False, go_backwards=True, input_shape=C_right.shape[1:]))
        
        merged = Merge([left_context, right_context], mode='concat',concat_axis=1)

        rnn_n_dude = Sequential()
        rnn_n_dude.add(merged)
        rnn_n_dude.add(Dense(40, init='he_normal'))
        rnn_n_dude.add(Activation('relu'))
        rnn_n_dude.add(Dense(40, init='he_normal'))
        rnn_n_dude.add(Activation('relu'))
        rnn_n_dude.add(Dense(3, init='he_normal'))
        rnn_n_dude.add(Activation('softmax'))
        rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        adagrad=Adagrad(clipnorm=1.5)
        adam=Adam()
        adadelta=Adadelta()
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        rnn_n_dude.compile(loss='poisson', optimizer=adam)
        rnn_n_dude.fit([C_left, C_right],Y,nb_epoch=10,batch_size=256,show_accuracy=True, verbose=0)
        
        # -----------------------------------------------------
        
        pred_class=rnn_n_dude.predict_classes([C_left, C_right], batch_size=256, verbose=0)
        s_nn_hat=hstack((zeros(k),pred_class,zeros(k)))
        x_nn_hat=N_DUDE.denoise_with_s(z[i],s_nn_hat,k)
        error_nn=N_DUDE.error_rate(x,x_nn_hat)
        
        print '1-D N-DUDE_RNN=', error_nn
        
        Error_One_NN_DUDE[i,k]=error_nn
        X_hat_One_NN_DUDE[k_max*i+k-1,:]=x_nn_hat
        
        s_class=3
        s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))
        Est_Loss_One_NN_DUDE[i,k]=est_loss_nn_dude
        
        ### 1-D N-DUDE Bound ###
        L_lower=np.array([[1-delta[i],1,0],[1-delta[i],1,0],[1-delta[i],0,1],[1-delta[i],0,1]])
        Y_lower = N_DUDE.make_data_for_One_NN_DUDE_LB(Z[i*n:(i+1)*n],k,L_lower,x,z[i],nb_classes,n)
    
        left_context = Sequential()
        left_context.add(LSTM(dim,init='he_normal',return_sequences=False, input_shape=C_left.shape[1:]))
        right_context = Sequential()
        right_context.add(LSTM(dim,init='he_normal',return_sequences=False, go_backwards=True, input_shape=C_right.shape[1:]))
        
        merged = Merge([left_context, right_context], mode='concat',concat_axis=1)

        rnn_n_dude = Sequential()
        rnn_n_dude.add(merged)
        rnn_n_dude.add(Dense(40, init='he_normal'))
        rnn_n_dude.add(Activation('relu'))
        rnn_n_dude.add(Dense(40, init='he_normal'))
        rnn_n_dude.add(Activation('relu'))
        rnn_n_dude.add(Dense(3, init='he_normal'))
        rnn_n_dude.add(Activation('softmax'))
        
        rnn_n_dude.compile(loss='poisson', optimizer=adam)
        rnn_n_dude.fit([C_left, C_right],Y_lower,nb_epoch=10,batch_size=256,show_accuracy=True, verbose=0)
    
        # -----------------------------------------------------
    
        pred_class_lower=rnn_n_dude.predict_classes([C_left, C_right], batch_size=256, verbose=0)
        s_nn_hat_lower=hstack((zeros(k),pred_class_lower,zeros(k)))
        x_nn_hat_lower=N_DUDE.denoise_with_s(z[i],s_nn_hat_lower,k)
        error_nn_lower=N_DUDE.error_rate(x,x_nn_hat_lower)
    
        print '1-D N-DUDE_RNN_Bound=', error_nn_lower
        Error_One_NN_DUDE_LB[i,k]=error_nn_lower
    
        X_hat_One_NN_DUDE_LB[k_max*i+k-1,:]=x_nn_hat_lower
        
        res_file='/home/ubuntu/Yoon_ICE/Results_Plot/Neural_DUDE_RNN_One_output_dim_%d_0.01'%dim
        np.savez(res_file,Error_One_NN_DUDE=Error_One_NN_DUDE, Error_One_NN_DUDE_LB=Error_One_NN_DUDE_LB, 
                 Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE, 
                 X_hat_One_NN_DUDE=X_hat_One_NN_DUDE, X_hat_One_NN_DUDE_LB=X_hat_One_NN_DUDE_LB)