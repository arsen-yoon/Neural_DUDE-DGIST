import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN, device=cpu,floatX=float32"
import theano
import keras
import time

import numpy as np
import Binary_DUDE as DUDE
import Binary_N_DUDE as N_DUDE

from numpy import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.utils import np_utils

### Pre-Generated Data Load ###
data=np.load('/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Data_Generation&Save/Neural_dude_Data_lena512.npz')

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
Error_One_DUDE=zeros((len(delta),k_max+1))
Error_Two_DUDE=Error_One_DUDE.copy()

Error_One_NN_DUDE=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD=Error_One_DUDE.copy()
Error_Two_NN_DUDE=Error_One_DUDE.copy()
Error_Two_NN_DUDE_PD=Error_One_DUDE.copy()

Error_One_NN_DUDE_LB=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD_LB=Error_One_DUDE.copy()
Error_Two_NN_DUDE_LB=Error_One_DUDE.copy()
Error_Two_NN_DUDE_PD_LB=Error_One_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_One_DUDE[i,0]=delta[i]
    Error_Two_DUDE[i,0]=delta[i]
    
    Error_One_NN_DUDE[i,0]=delta[i]
    Error_One_NN_DUDE_PD[i,0]=delta[i]
    Error_Two_NN_DUDE[i,0]=delta[i]
    Error_Two_NN_DUDE_PD[i,0]=delta[i]
    
    Error_One_NN_DUDE_LB[i,0]=delta[i]
    Error_One_NN_DUDE_PD_LB[i,0]=delta[i]
    Error_Two_NN_DUDE_LB[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_LB[i,0]=delta[i]

### Est Loss Mat ###
Est_Loss_One_NN_DUDE=zeros((len(delta),k_max+1))
Est_Loss_One_NN_DUDE_PD=Est_Loss_One_NN_DUDE.copy()
Est_Loss_Two_NN_DUDE=Est_Loss_One_NN_DUDE.copy()
Est_Loss_Two_NN_DUDE_PD=Est_Loss_One_NN_DUDE.copy()

for i in range(len(delta)):
    Est_Loss_One_NN_DUDE[i,0]=delta[i] # 1-D N-DUDE
    Est_Loss_One_NN_DUDE_PD[i,0]=delta[i] # 1-D N-DUDE Padding
    Est_Loss_Two_NN_DUDE[i,0]=delta[i] # 2-D N-DUDE
    Est_Loss_Two_NN_DUDE_PD[i,0]=delta[i] # 2-D N-DUDE Padding
    
### X_hat Mat ###
X_hat_One_DUDE=zeros((len(delta)*k_max,n))
X_hat_Two_DUDE=X_hat_One_DUDE.copy()

X_hat_One_NN_DUDE=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD=X_hat_One_DUDE.copy()

X_hat_Two_NN_DUDE=X_hat_One_DUDE.copy()
X_hat_Two_NN_DUDE_PD=X_hat_One_DUDE.copy()

X_hat_One_NN_DUDE_LB=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD_LB=X_hat_One_DUDE.copy()
X_hat_Two_NN_DUDE_LB=X_hat_One_DUDE.copy()
X_hat_Two_NN_DUDE_PD_LB=X_hat_One_DUDE.copy()

for i in range(len(delta)):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        Total_Start=time.time()
        ### 1-D N-DUDE ###
        C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
        model=Sequential()
        model.add(Dense(3,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('softmax'))
        
        rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        adagrad=Adagrad(clipnorm=1.5)
        adam=Adam()
        adadelta=Adadelta()
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C,Y,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C, Y))
        
        # -----------------------------------------------------
        
        pred_class=model.predict_classes(C, batch_size=200, verbose=0)
        s_nn_hat=hstack((zeros(k),pred_class,zeros(k)))
        x_nn_hat=N_DUDE.denoise_with_s(z[i],s_nn_hat,k)
        error_nn=N_DUDE.error_rate(x,x_nn_hat)
        
        print '1-D N-DUDE=', error_nn
        
        Error_One_NN_DUDE[i,k]=error_nn
        X_hat_One_NN_DUDE[k_max*i+k-1,:]=x_nn_hat
        
        s_class=3
        s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))
        Est_Loss_One_NN_DUDE[i,k]=est_loss_nn_dude
        
        ### 1-D N-DUDE Padding ###
        C_PD,Y_PD = N_DUDE.make_data_for_One_NN_DUDE_PD(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
        model=Sequential()
        model.add(Dense(3,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_PD,Y_PD,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_PD, Y_PD))
        
        # -----------------------------------------------------
        
        s_nn_hat_PD=model.predict_classes(C_PD, batch_size=200, verbose=0)
        x_nn_hat_PD=N_DUDE.denoise_with_s_One_NN_PD(z[i],s_nn_hat_PD)
        error_nn_PD=N_DUDE.error_rate(x,x_nn_hat_PD)
        
        print '1-D N-DUDE Padding=', error_nn_PD
        
        Error_One_NN_DUDE_PD[i,k]=error_nn_PD
        X_hat_One_NN_DUDE_PD[k_max*i+k-1,:]=x_nn_hat_PD
        
        s_class=3
        s_nn_hat_cat_PD=np_utils.to_categorical(s_nn_hat_PD,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_PD=mean(sum(emp_dist*s_nn_hat_cat_PD,axis=1))
        Est_Loss_One_NN_DUDE_PD[i,k]=est_loss_nn_dude_PD
        
        Total_End=time.time()
        
        print ""
        print 'Total_Time=', Total_End-Total_Start
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Regression_One'
        np.savez(res_file,Error_One_DUDE=Error_One_DUDE,
                 Error_One_NN_DUDE=Error_One_NN_DUDE,Error_One_NN_DUDE_PD=Error_One_NN_DUDE_PD,
                 Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE, Est_Loss_One_NN_DUDE_PD=Est_Loss_One_NN_DUDE_PD,
                X_hat_One_NN_DUDE=X_hat_One_NN_DUDE, X_hat_One_NN_DUDE_PD=X_hat_One_NN_DUDE_PD)