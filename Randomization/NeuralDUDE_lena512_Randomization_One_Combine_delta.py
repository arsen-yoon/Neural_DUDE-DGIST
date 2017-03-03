import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN, device=cpu,floatX=float32"
import theano
import keras
import time

import numpy as np
import Binary_DUDE as DUDE
import Binary_N_DUDE as N_DUDE
import Randomization as RD

from numpy import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.utils import np_utils

Total_Start=time.time()
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
Error_One_NN_DUDE_RD=zeros((len(delta),k_max+1))
Error_One_NN_DUDE_Bind_RD=Error_One_NN_DUDE_RD.copy()
Error_One_NN_DUDE_Bind_Norm_RD=Error_One_NN_DUDE_RD.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_One_NN_DUDE_RD[i,0]=delta[i]
    Error_One_NN_DUDE_Bind_RD[i,0]=delta[i]
    Error_One_NN_DUDE_Bind_Norm_RD[i,0]=delta[i]
    
### Est Loss Mat ###
for i in range(len(delta)):
    Est_Loss_NN_DUDE_RD=zeros((len(delta),k_max+1))
    Est_Loss_NN_DUDE_RD[i,0]=delta[i] # 1-D N-DUDE
    Est_Loss_NN_DUDE_Bind_RD=zeros((len(delta),k_max))
    Est_Loss_NN_DUDE_Bind_RD[i,0]=delta[i] # 1-D N-DUDE Bind
    Est_Loss_NN_DUDE_Bind_Norm_RD=zeros((len(delta),k_max))
    Est_Loss_NN_DUDE_Bind_Norm_RD[i,0]=delta[i] # 1-D N-DUDE Bind Norm
    
### X_hat Mat ###
X_hat_One_NN_DUDE_RD=zeros((len(delta)*k_max,n))
X_hat_One_NN_DUDE_Bind_RD=X_hat_One_NN_DUDE_RD.copy()
X_hat_One_NN_DUDE_Bind_Norm_RD=X_hat_One_NN_DUDE_RD.copy()

for i in range(9,10):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        One_Start=time.time()

        ### 1-D N-DUDE Context Bind ###
    
        C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
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
        model.fit(C_Bind,Y_Bind,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind))
        
        # -----------------------------------------------------
        
        proba_bind=model.predict_proba(C,batch_size=200,verbose=0)
        proba_bind_new=RD.rand_new(proba_bind,delta[i])
        class_bind_new=RD.make_new_class(proba_bind_new)
        s_nn_hat_Bind=hstack((zeros(k),class_bind_new,zeros(k)))
        x_nn_hat_Bind=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind,k)
        error_nn_Bind=N_DUDE.error_rate(x,x_nn_hat_Bind)
        
        print '1-D N-DUDE Context Bind=', error_nn_Bind
        
        Error_One_NN_DUDE_Bind_RD[i,k]=error_nn_Bind
        X_hat_One_NN_DUDE_Bind_RD[k_max*i+k-1,:]=x_nn_hat_Bind
        
        s_class=3
        s_nn_hat_cat_Bind=np_utils.to_categorical(s_nn_hat_Bind,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_Bind=mean(sum(emp_dist*s_nn_hat_cat_Bind,axis=1))
        Est_Loss_NN_DUDE_RD[i,k]=est_loss_nn_dude_Bind
        
        ### 1-D N-DUDE Context Bind Normalization ###
        
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
        model.fit(C_Bind,Y_Bind_Norm,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind_Norm))
        
        # -----------------------------------------------------
        
        proba_bind_norm=model.predict_proba(C,batch_size=200,verbose=0)
        proba_bind_norm_new=RD.rand_new(proba_bind_norm,delta[i])
        class_bind_norm_new=RD.make_new_class(proba_bind_norm_new)
        s_nn_hat_Bind_Norm=hstack((zeros(k),class_bind_norm_new,zeros(k)))
        x_nn_hat_Bind_Norm=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind_Norm,k)
        error_nn_Bind_Norm=N_DUDE.error_rate(x,x_nn_hat_Bind_Norm)
        
        print '1-D N-DUDE Context Bind Normalization=', error_nn_Bind_Norm
        
        Error_One_NN_DUDE_Bind_Norm_RD[i,k]=error_nn_Bind_Norm
        X_hat_One_NN_DUDE_Bind_Norm_RD[k_max*i+k-1,:]=x_nn_hat_Bind_Norm
        
        s_class=3
        s_nn_hat_cat_Bind_Norm=np_utils.to_categorical(s_nn_hat_Bind_Norm,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_Bind_Norm=mean(sum(emp_dist*s_nn_hat_cat_Bind_Norm,axis=1))
        Est_Loss_NN_DUDE_Bind_Norm_RD[i,k]=est_loss_nn_dude_Bind_Norm
        
        
        One_End=time.time()
        One=One_End-One_Start
        
        print ''
        print '*Time*'
        print 'Total=%0.1f'%One
        print '-----------------------------------------------------'
        
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Randomization_Combine_0.1'
        np.savez(res_file,Error_One_NN_DUDE_Bind_RD=Error_One_NN_DUDE_Bind_RD,Error_One_NN_DUDE_Bind_Norm_RD=Error_One_NN_DUDE_Bind_Norm_RD,X_hat_One_NN_DUDE_Bind_Norm_RD=X_hat_One_NN_DUDE_Bind_Norm_RD, X_hat_One_NN_DUDE_Bind_RD=X_hat_One_NN_DUDE_Bind_RD, Est_Loss_NN_DUDE_Bind_Norm_RD=Est_Loss_NN_DUDE_Bind_Norm_RD, Est_Loss_NN_DUDE_Bind_RD=Est_Loss_NN_DUDE_Bind_RD)

        Total_End=time.time()
Total=Total_End-Total_Start
print "I'm Done!", Total
