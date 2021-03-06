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

### Time Mat ###
num_dude=2
num_nn=8

DUDE_Time=zeros((len(delta)*(num_dude+1),k_max)) # for tot_time, +1
NN_Time=zeros((len(delta)*(num_nn+1),k_max))

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
        Total_DUDE_Start=time.time()
        One_DUDE_Start=time.time()
        
        ### 1-D DUDE ###
        s_hat,m=DUDE.One_DUDE(z[i],k,delta[i])
        x_dude_hat=DUDE.denoise_with_s(z[i],s_hat,k)
        error_dude=DUDE.error_rate(x,x_dude_hat)
        print '1-D DUDE=',error_dude
        
        Error_One_DUDE[i,k]=error_dude
        X_hat_One_DUDE[k_max*i+k-1,:]=x_dude_hat
        
        One_DUDE_End=time.time()
        One_DUDE_Duration=One_DUDE_End-One_DUDE_Start
        
        ### 2-D DUDE ###
        Two_DUDE_Start=time.time()
        s_hat_two,m=DUDE.Two_DUDE(z_two[i],k,delta[i],n,offset) 
        x_dude_hat_two=DUDE.denoise_with_s_Two_DUDE(z_two[i],s_hat_two,k)
        error_dude_two=DUDE.error_rate(x,x_dude_hat_two)
        print '2-D DUDE=',error_dude_two
        
        Error_Two_DUDE[i,k]=error_dude_two
        X_hat_Two_DUDE[k_max*i+k-1,:]=x_dude_hat_two
        
        Two_DUDE_End=time.time()
        Two_DUDE_Duration=Two_DUDE_End-Two_DUDE_Start
        
        Total_DUDE_End=time.time()
        Total_DUDE_Duration=Total_DUDE_End-Total_DUDE_Start
        
        DUDE_Time[i*3,k-1]=One_DUDE_Duration
        DUDE_Time[i*3+1,k-1]=Two_DUDE_Duration
        DUDE_Time[i*3+2,k-1]=Total_DUDE_Duration
        
        ### 1-D N-DUDE ###
        Total_NN_Start=time.time()
        One_NN_Start=time.time()
        C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
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
        
        One_NN_End=time.time()
        One_NN_Duration=One_NN_End-One_NN_Start
        
        ### 1-D N-DUDE Padding ###
        One_NN_PD_Start=time.time()
        C_PD,Y_PD = N_DUDE.make_data_for_One_NN_DUDE_PD(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
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
        
        One_NN_PD_End=time.time()
        One_NN_PD_Duration=One_NN_PD_End-One_NN_PD_Start
        
        ### 2-D N-DUDE ###
        Two_NN_Start=time.time()
        C_two,Y_two = N_DUDE.make_data_for_Two_NN_DUDE(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],
                                                       nb_classes,n,offset)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two, Y_two))
        
        # -----------------------------------------------------
    
        pred_class_two=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE=', error_nn_two
        
        Error_Two_NN_DUDE[i,k]=error_nn_two
        X_hat_Two_NN_DUDE[k_max*i+k-1,:]=x_nn_hat_two
           
        s_class_two=3
        s_nn_hat_cat_two=np_utils.to_categorical(s_nn_hat_two,s_class_two)
        emp_dist_two=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two=mean(sum(emp_dist_two*s_nn_hat_cat_two,axis=1))
        Est_Loss_Two_NN_DUDE[i,k]=est_loss_nn_dude_two
        
        Two_NN_End=time.time()
        Two_NN_Duration=Two_NN_End-Two_NN_Start
        
        ### 2-D N-DUDE Padding ###
        Two_NN_PD_Start=time.time()
        C_two_padding,Y_two_padding = N_DUDE.make_data_for_Two_NN_DUDE_PD(P_padding[i],P[i],k,
                                                                          L_new[i*alpha_size:(i+1)*alpha_size,],
                                                                          nb_classes,n,offset,dp)
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_padding,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_padding))
        
        # -----------------------------------------------------
    
        s_nn_hat_two_padding=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_padding=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two_padding) 
        error_nn_two_padding=N_DUDE.error_rate(x,x_nn_hat_two_padding)
        print '2-D N-DUDE_Padding=', error_nn_two_padding
        
        Error_Two_NN_DUDE_PD[i,k]=error_nn_two_padding
        X_hat_Two_NN_DUDE_PD[k_max*i+k-1,:]=x_nn_hat_two_padding
       
        s_class_two_padding=3
        s_nn_hat_cat_two_padding=np_utils.to_categorical(s_nn_hat_two_padding,s_class_two_padding)
        emp_dist_two_padding=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two_padding=mean(sum(emp_dist_two_padding*s_nn_hat_cat_two_padding,axis=1))
        Est_Loss_Two_NN_DUDE_PD[i,k]=est_loss_nn_dude_two_padding
        
        Two_NN_PD_End=time.time()
        Two_NN_PD_Duration=Two_NN_PD_End-Two_NN_PD_Start
        
        ### 1-D N-DUDE Bound ###
        One_NN_LB_Start=time.time()
        Y_lower = N_DUDE.make_data_for_One_NN_DUDE_LB(Z[i*n:(i+1)*n],k,L_lower,x,z[i],nb_classes,n)
    
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C,Y_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C, Y_lower))
    
        # -----------------------------------------------------
    
        pred_class_lower=model.predict_classes(C, batch_size=200, verbose=0)
        s_nn_hat_lower=hstack((zeros(k),pred_class_lower,zeros(k)))
        x_nn_hat_lower=N_DUDE.denoise_with_s(z[i],s_nn_hat_lower,k)
        error_nn_lower=N_DUDE.error_rate(x,x_nn_hat_lower)
    
        print '1-D N-DUDE_Bound=', error_nn_lower
        Error_One_NN_DUDE_LB[i,k]=error_nn_lower
    
        X_hat_One_NN_DUDE_LB[k_max*i+k-1,:]=x_nn_hat_lower
        One_NN_LB_End=time.time()
        One_NN_LB_Duration=One_NN_LB_End-One_NN_LB_Start
        
        ### 1-D N-DUDE Padding Bound ###
        One_NN_PD_LB_Start=time.time()
        Y_PD_lower = N_DUDE.make_data_for_One_NN_DUDE_PD_LB(P[i],Z[i*n:(i+1)*n],im_bin,k,L_lower,nb_classes,n)
    
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_PD,Y_PD_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_PD, Y_PD_lower))
    
        # -----------------------------------------------------
    
        s_nn_hat_PD_lower=model.predict_classes(C_PD, batch_size=200, verbose=0)
        x_nn_hat_PD_lower=N_DUDE.denoise_with_s_One_NN_PD(z[i],s_nn_hat_PD_lower)
        error_nn_PD_lower=N_DUDE.error_rate(x,x_nn_hat_PD_lower)
    
        print '1-D N-DUDE Padding Bound=', error_nn_PD_lower
        Error_One_NN_DUDE_PD_LB[i,k]=error_nn_PD_lower
    
        X_hat_One_NN_DUDE_PD_LB[k_max*i+k-1,:]=x_nn_hat_PD_lower
        One_NN_PD_LB_End=time.time()
        One_NN_PD_LB_Duration=One_NN_PD_LB_End-One_NN_PD_LB_Start

        ### 2-D N-DUDE Bound ###
        Two_NN_LB_Start=time.time()
        Y_two_lower = N_DUDE.make_data_for_Two_NN_DUDE_LB(P[i],k,L_lower,im_bin,nb_classes,
                                                                      n,offset)

        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two, Y_two_lower))
    
        # -----------------------------------------------------
    
        pred_class_two_lower=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two_lower=N_DUDE.mapping_mat_resize(pred_class_two_lower,k,n)
        x_nn_hat_two_lower=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two_lower)
        error_nn_two_lower=N_DUDE.error_rate(x,x_nn_hat_two_lower)
    
        print '2-D N-DUDE_Bound=', error_nn_two_lower
        Error_Two_NN_DUDE_LB[i,k]=error_nn_two_lower
        X_hat_Two_NN_DUDE_LB[k_max*i+k-1,:]=x_nn_hat_two_lower
        
        Two_NN_LB_End=time.time()
        Two_NN_LB_Duration=Two_NN_LB_End-Two_NN_LB_Start
        
        ### 2-D N-DUDE Padding Bound ###
        Two_NN_PD_LB_Start=time.time()
        Y_two_PD_lower = N_DUDE.make_data_for_Two_NN_DUDE_PD_LB(P_padding[i],P[i],im_bin,k,L_lower,nb_classes,n,offset,dp)

        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_PD_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_PD_lower))
    
        # -----------------------------------------------------
    
        s_nn_hat_two_PD_lower=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_PD_lower=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two_PD_lower)
        error_nn_two_PD_lower=N_DUDE.error_rate(x,x_nn_hat_two_PD_lower)
    
        print '2-D N-DUDE_PD_Bound=', error_nn_two_PD_lower
        Error_Two_NN_DUDE_PD_LB[i,k]=error_nn_two_PD_lower
        X_hat_Two_NN_DUDE_PD_LB[k_max*i+k-1,:]=x_nn_hat_two_PD_lower
        
        Two_NN_PD_LB_End=time.time()
        Two_NN_PD_LB_Duration=Two_NN_PD_LB_End-Two_NN_PD_LB_Start
        
        Total_NN_End=time.time()
        Total_NN_Duration=Total_NN_End-Total_NN_Start
        
        NN_Time[i*6,k-1]=One_NN_Duration
        NN_Time[i*6+1,k-1]=One_NN_LB_Duration
        NN_Time[i*6+2,k-1]=One_NN_PD_Duration
        NN_Time[i*6+3,k-1]=One_NN_PD_LB_Duration
        NN_Time[i*6+4,k-1]=Two_NN_Duration
        NN_Time[i*6+5,k-1]=Two_NN_LB_Duration
        NN_Time[i*6+6,k-1]=Two_NN_PD_Duration
        NN_Time[i*6+7,k-1]=Two_NN_PD_LB_Duration
        NN_Time[i*6+8,k-1]=Total_NN_Duration
        
        
        print ""
        print "1-D_DUDE time=",One_DUDE_Duration, "2-D_DUDE time=",Two_DUDE_Duration
        print "Total_DUDE time=",Total_DUDE_Duration
        print '1-D_NN_Time=', One_NN_Duration, '1-D_NN_PD_Time=', One_NN_PD_Duration, '2-D_NN_time=', Two_NN_Duration, '2-D_NN_PD_Time=', Two_NN_PD_Duration 
        print '1-D_NN_Bound_Time=', One_NN_LB_Duration, '1-D_NN_PD_Bound_Time=', One_NN_PD_LB_Duration,'2-D_NN_Bound_Time=', Two_NN_LB_Duration, '2-D_NN_PD_Bound_Time=', Two_NN_PD_LB_Duration
        print 'Total_NN_Time=', Total_NN_Duration
        print 'Total_Time=', Total_NN_Duration+Total_DUDE_Duration
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Neural_DUDE_Result_lena512_Basics'
        np.savez(res_file,Error_One_DUDE=Error_One_DUDE, Error_Two_DUDE=Error_Two_DUDE, Error_One_NN_DUDE=Error_One_NN_DUDE, Error_One_NN_DUDE_PD=Error_One_NN_DUDE_PD, Error_One_NN_DUDE_PD_LB=Error_One_NN_DUDE_PD_LB,
                 Error_One_NN_DUDE_LB=Error_One_NN_DUDE_LB, Error_Two_NN_DUDE=Error_Two_NN_DUDE,
                 Error_Two_NN_DUDE_LB=Error_Two_NN_DUDE_LB, Error_Two_NN_DUDE_PD=Error_Two_NN_DUDE_PD,
                 Error_Two_NN_DUDE_PD_LB=Error_Two_NN_DUDE_PD_LB,
                 Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE, Est_Loss_One_NN_DUDE_PD=Est_Loss_One_NN_DUDE_PD,Est_Loss_Two_NN_DUDE=Est_Loss_Two_NN_DUDE, Est_Loss_Two_NN_DUDE_PD=Est_Loss_Two_NN_DUDE_PD,DUDE_Time=DUDE_Time, NN_Time=NN_Time)