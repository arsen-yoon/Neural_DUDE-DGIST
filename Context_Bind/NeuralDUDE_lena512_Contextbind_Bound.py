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
Error_One_NN_DUDE_Bind_Bound=zeros((len(delta),k_max+1))
Error_One_NN_DUDE_Bind_Norm_Bound=Error_One_NN_DUDE_Bind_Bound.copy()

num_nn=2
NN_Time=zeros((len(delta)*(num_nn+1),k_max))
Hist_Bind_Bound=zeros((k_max*4,10))
Hist_Bind_Norm_Bound=Hist_Bind_Bound.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_One_NN_DUDE_Bind_Bound[i,0]=delta[i]
    Error_One_NN_DUDE_Bind_Norm_Bound[i,0]=delta[i]
    
### X_hat Mat ###
X_hat_One_NN_DUDE_Bind_Bound=np.zeros((len(delta)*k_max,n))
X_hat_One_NN_DUDE_Bind_Norm_Bound=X_hat_One_NN_DUDE_Bind_Bound.copy()

for i in range(len(delta)):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        ### 1-D N-DUDE Context Bind Bound ###
        One_NN_Bind_Bound_Start=time.time()
        C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,
                                               L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        C_Bind_Bound,Y_Bind_Bound,Y_Bind_Norm_Bound,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind_LB(x,z[i],Z[i*n:(i+1)*n,],k,L_lower,nb_classes,n)
            
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
        hist=model.fit(C_Bind_Bound,Y_Bind_Bound,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind_Bound, Y_Bind_Bound))
        Hist_Bind_Bound[4*k-4]=hist.history['acc']
        Hist_Bind_Bound[4*k-3]=hist.history['loss']
        Hist_Bind_Bound[4*k-2]=hist.history['val_acc']
        Hist_Bind_Bound[4*k-1]=hist.history['val_loss']
        # -----------------------------------------------------
        
        pred_class_Bind_Bound=model.predict_classes(C,batch_size=200,verbose=0)
        s_nn_hat_Bind_Bound=hstack((zeros(k),pred_class_Bind_Bound,zeros(k)))
        x_nn_hat_Bind_Bound=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind_Bound,k)
        error_nn_Bind_Bound=N_DUDE.error_rate(x,x_nn_hat_Bind_Bound)
        
        print '1-D N-DUDE Context Bind Bound=', error_nn_Bind_Bound
        
        Error_One_NN_DUDE_Bind_Bound[i,k]=error_nn_Bind_Bound
        X_hat_One_NN_DUDE_Bind_Bound[i*k_max+k-1]=x_nn_hat_Bind_Bound
        
        One_NN_Bind_Bound_End=time.time()
        One_NN_Bind_Bound_Duration=One_NN_Bind_Bound_End-One_NN_Bind_Bound_Start
        ### 1-D N-DUDE Context Bind Normalization Bound ###
        One_NN_Bind_Norm_Bound_Start=time.time()
        
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
        hist=model.fit(C_Bind_Bound,Y_Bind_Norm_Bound,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind_Bound, Y_Bind_Norm_Bound))
        Hist_Bind_Norm_Bound[4*k-4]=hist.history['acc']
        Hist_Bind_Norm_Bound[4*k-3]=hist.history['loss']
        Hist_Bind_Norm_Bound[4*k-2]=hist.history['val_acc']
        Hist_Bind_Norm_Bound[4*k-1]=hist.history['val_loss']
        # -----------------------------------------------------
        
        pred_class_Bind_Norm_Bound=model.predict_classes(C,batch_size=200,verbose=0)
        s_nn_hat_Bind_Norm_Bound=hstack((zeros(k),pred_class_Bind_Norm_Bound,zeros(k)))
        x_nn_hat_Bind_Norm_Bound=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind_Norm_Bound,k)
        error_nn_Bind_Norm_Bound=N_DUDE.error_rate(x,x_nn_hat_Bind_Norm_Bound)
        
        print '1-D N-DUDE Context Bind Normalization Bound=', error_nn_Bind_Norm_Bound
        
        Error_One_NN_DUDE_Bind_Norm_Bound[i,k]=error_nn_Bind_Norm_Bound
        X_hat_One_NN_DUDE_Bind_Norm_Bound[i*k_max+k-1]=x_nn_hat_Bind_Norm_Bound
        
        One_NN_Bind_Norm_Bound_End=time.time()
        One_NN_Bind_Norm_Bound_Duration=One_NN_Bind_Norm_Bound_End-One_NN_Bind_Norm_Bound_Start
        
        print ''
        print '1-D N-DUDE CB=%0.1f'%One_NN_Bind_Bound_Duration, '1-D N-DUDE CB Norm=%0.1f'%One_NN_Bind_Norm_Bound_Duration
        print 'Total Time=', One_NN_Bind_Bound_Duration+One_NN_Bind_Norm_Bound_Duration
        print '-----------------------------------------------------'
        
        NN_Time[i*3,k-1]=One_NN_Bind_Bound_Duration
        NN_Time[i*3+1,k-1]=One_NN_Bind_Norm_Bound_Duration
        NN_Time[i*3+2,k-1]=One_NN_Bind_Bound_Duration+One_NN_Bind_Norm_Bound_Duration
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Neural_DUDE_Result_lena512_Combined&Norm_Bound'
        np.savez(res_file,Error_One_NN_DUDE_Bind_Bound=Error_One_NN_DUDE_Bind_Bound,
                 Error_One_NN_DUDE_Bind_Norm_Bound=Error_One_NN_DUDE_Bind_Norm_Bound,
                 Hist_Bind_Bound=Hist_Bind_Bound, Hist_Bind_Norm_Bound=Hist_Bind_Norm_Bound, NN_Time=NN_Time)

        Total_End=time.time()
Total=Total_End-Total_Start
print "I'm Done in %0.2fsecs!"%Total
