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
P_padding=np.zeros((len(delta), imarray.shape[0]+2*loss_lines, imarray.shape[1]+2*loss_lines, nb_classes),dtype=np.int)

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
Error_One_NN_DUDE_Bind=zeros((len(delta),k_max+1))
Error_One_NN_DUDE_Bind_Norm=Error_One_NN_DUDE_Bind.copy()

num_nn=2
NN_Time=zeros((len(delta)*(num_nn+1),k_max))
Hist_Bind=zeros((k_max*4,10))
Hist_Bind_Norm=Hist_Bind.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_One_NN_DUDE_Bind[i,0]=delta[i]
    Error_One_NN_DUDE_Bind_Norm[i,0]=delta[i]
    
### Est Loss Mat ###
Est_Loss_NN_DUDE_CB=zeros((len(delta),k_max+1))
for i in range(len(delta)):
    Est_Loss_NN_DUDE_CB[i,0]=delta[i] # 1-D N-DUDE Bind
    Est_Loss_NN_DUDE_Norm=Est_Loss_NN_DUDE_CB.copy() # 1-D N-DUDE Bind Norm
    
### X_hat Mat ###
X_hat_One_NN_DUDE_Bind=np.zeros((len(delta)*k_max,n))
X_hat_One_NN_DUDE_Bind_Norm=X_hat_One_NN_DUDE_Bind.copy()

for i in range(9,10):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        ### 1-D N-DUDE Context Bind ###
        One_NN_Bind_Start=time.time()
        C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
            
        model=Sequential()
        model.add(Dense(3,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('softmax'))
        
        rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        adagrad=Adagrad(clipnorm=1.5)
        adam=Adam()
        adadelta=Adadelta()
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        model.compile(loss='poisson', optimizer=adam)
        hist=model.fit(C_Bind,Y_Bind,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind))
        Hist_Bind[4*k-4]=hist.history['acc']
        Hist_Bind[4*k-3]=hist.history['loss']
        Hist_Bind[4*k-2]=hist.history['val_acc']
        Hist_Bind[4*k-1]=hist.history['val_loss']
        # -----------------------------------------------------
        
        pred_class_Bind=model.predict_classes(C,batch_size=200,verbose=0)
        s_nn_hat_Bind=hstack((zeros(k),pred_class_Bind,zeros(k)))
        x_nn_hat_Bind=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind,k)
        error_nn_Bind=N_DUDE.error_rate(x,x_nn_hat_Bind)
        
        print '1-D N-DUDE Context Bind=', error_nn_Bind
        
        Error_One_NN_DUDE_Bind[i,k]=error_nn_Bind
        X_hat_One_NN_DUDE_Bind[i*k_max+k-1]=x_nn_hat_Bind
        
        s_class=3
        s_nn_hat_cat_Bind=np_utils.to_categorical(s_nn_hat_Bind,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_Bind=mean(sum(emp_dist*s_nn_hat_cat_Bind,axis=1))
        Est_Loss_NN_DUDE_CB[i,k]=est_loss_nn_dude_Bind
        
        One_NN_Bind_End=time.time()
        One_NN_Bind_Duration=One_NN_Bind_End-One_NN_Bind_Start
        
        ### 1-D N-DUDE Context Bind Normalization ###
        One_NN_Bind_Norm_Start=time.time()
        
        model=Sequential()
        model.add(Dense(3,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        hist=model.fit(C_Bind,Y_Bind_Norm,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind_Norm))
        Hist_Bind_Norm[4*k-4]=hist.history['acc']
        Hist_Bind_Norm[4*k-3]=hist.history['loss']
        Hist_Bind_Norm[4*k-2]=hist.history['val_acc']
        Hist_Bind_Norm[4*k-1]=hist.history['val_loss']
        # -----------------------------------------------------
        
        pred_class_Bind_Norm=model.predict_classes(C,batch_size=200,verbose=0)
        s_nn_hat_Bind_Norm=hstack((zeros(k),pred_class_Bind_Norm,zeros(k)))
        x_nn_hat_Bind_Norm=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind_Norm,k)
        error_nn_Bind_Norm=N_DUDE.error_rate(x,x_nn_hat_Bind_Norm)
        
        print '1-D N-DUDE Context Bind Normalization=', error_nn_Bind_Norm
        
        Error_One_NN_DUDE_Bind_Norm[i,k]=error_nn_Bind_Norm
        X_hat_One_NN_DUDE_Bind_Norm[i*k_max+k-1]=x_nn_hat_Bind_Norm
        
        s_class=3
        s_nn_hat_cat_Bind_Norm=np_utils.to_categorical(s_nn_hat_Bind_Norm,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_Bind_Norm=mean(sum(emp_dist*s_nn_hat_cat_Bind_Norm,axis=1))
        Est_Loss_NN_DUDE_Norm[i,k]=est_loss_nn_dude_Bind_Norm
        
        One_NN_Bind_Norm_End=time.time()
        One_NN_Bind_Norm_Duration=One_NN_Bind_Norm_End-One_NN_Bind_Norm_Start
        
        print ''
        print '1-D N-DUDE CB=%0.1f'%One_NN_Bind_Duration, '1-D N-DUDE CB Norm=%0.1f'%One_NN_Bind_Norm_Duration
        print 'Total Time=', One_NN_Bind_Duration+One_NN_Bind_Norm_Duration
        print '-----------------------------------------------------'
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Regression_Combine&Norm_0.1'
        np.savez(res_file,Error_One_NN_DUDE_Bind=Error_One_NN_DUDE_Bind,
                 Error_One_NN_DUDE_Bind_Norm=Error_One_NN_DUDE_Bind_Norm,
                 Est_Loss_NN_DUDE_CB=Est_Loss_NN_DUDE_CB, Est_Loss_NN_DUDE_Norm=Est_Loss_NN_DUDE_Norm,
                 Hist_Bind=Hist_Bind, Hist_Bind_Norm=Hist_Bind_Norm)

        Total_End=time.time()
Total=Total_End-Total_Start
print "I'm Done in %0.2fsecs!"%Total
