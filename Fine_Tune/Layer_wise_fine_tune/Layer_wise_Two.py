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
from keras.models import model_from_json
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
Error_Two_DUDE=zeros((len(delta),k_max+1))

Error_Two_NN_DUDE=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_new=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD_new=Error_Two_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_Two_DUDE[i,0]=delta[i]
    
    Error_Two_NN_DUDE[i,0]=delta[i]
    Error_Two_NN_DUDE_new[i,0]=delta[i]
    Error_Two_NN_DUDE_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE_PD[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_new[i,0]=delta[i]
    
### Est Loss Mat ###
Est_Loss_Two_NN_DUDE=zeros((len(delta),k_max+1))
Est_Loss_Two_NN_DUDE_new=zeros((len(delta),k_max+1))
Est_Loss_Two_NN_DUDE_Pre=zeros((len(delta),k_max+1))
Est_Loss_Two_NN_DUDE_PD=Est_Loss_Two_NN_DUDE.copy()
Est_Loss_Two_NN_DUDE_PD_Pre=Est_Loss_Two_NN_DUDE.copy()
Est_Loss_Two_NN_DUDE_PD_new=Est_Loss_Two_NN_DUDE.copy()

for i in range(len(delta)):
    Est_Loss_Two_NN_DUDE[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_new[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_Pre[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_PD[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_PD_Pre[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_PD_new[i,0]=delta[i] 

### X_hat Mat ###
X_hat_Two_DUDE=zeros((len(delta)*k_max,n))
X_hat_Two_NN_DUDE_new=zeros((len(delta)*k_max,n))
X_hat_Two_NN_DUDE=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD_new=X_hat_Two_DUDE.copy()

for i in range(0,1):
    print "##### delta=%0.2f #####" % delta[i]
    for k in range(1,k_max+1):
        print 'k=',k
        Total_Start=time.time()
        
        ### 2-D DUDE ###
        Two_DUDE_Start=time.time()
        s_hat_two,m=DUDE.Two_DUDE(z_two[i],k,delta[i],n,offset) 
        x_dude_hat_two=DUDE.denoise_with_s_Two_DUDE(z_two[i],s_hat_two,k)
        error_dude_two=DUDE.error_rate(x,x_dude_hat_two)
        print '2-D DUDE=',error_dude_two
        Error_Two_DUDE[i,k]=error_dude_two
        X_hat_Two_DUDE[k_max*i+k-1,:]=x_dude_hat_two
        
        ### 2-D N-DUDE ###
        C_two,Y_two = N_DUDE.make_data_for_Two_NN_DUDE(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n,offset)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        
        rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        adagrad=Adagrad(clipnorm=1.5)
        adam=Adam()
        adadelta=Adadelta()
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two, Y_two))
        
        # -----------------------------------------------------
    
        pred_class_two=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE=', error_nn_two
        
        Error_Two_NN_DUDE_Pre[i,k]=error_nn_two
        X_hat_Two_NN_DUDE_Pre[k_max*i+k-1,:]=x_nn_hat_two
           
        s_class_two=3
        s_nn_hat_cat_two=np_utils.to_categorical(s_nn_hat_two,s_class_two)
        emp_dist_two=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two=mean(sum(emp_dist_two*s_nn_hat_cat_two,axis=1))
        Est_Loss_Two_NN_DUDE_Pre[i,k]=est_loss_nn_dude_two
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal', trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        tmp=model.get_weights()
        tmp[0]=saved[0]
        tmp[1]=saved[1] 
        model.set_weights(tmp) # set new weights 
        
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two, Y_two))
        
        # -----------------------------------------------------
    
        pred_class_two=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE =', error_nn_two
        
        Error_Two_NN_DUDE[i,k]=error_nn_two
        X_hat_Two_NN_DUDE[k_max*i+k-1,:]=x_nn_hat_two
           
        s_class_two=3
        s_nn_hat_cat_two=np_utils.to_categorical(s_nn_hat_two,s_class_two)
        emp_dist_two=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two=mean(sum(emp_dist_two*s_nn_hat_cat_two,axis=1))
        Est_Loss_Two_NN_DUDE[i,k]=est_loss_nn_dude_two
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal',trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        tmp=model.get_weights()
        tmp[0]=saved[0]
        tmp[1]=saved[1]
        tmp[2]=saved[2]
        tmp[3]=saved[3]
        model.set_weights(tmp) # set new weights 
        
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two, Y_two))
        
        # -----------------------------------------------------
    
        pred_class_two=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE =', error_nn_two
        
        Error_Two_NN_DUDE_new[i,k]=error_nn_two
        X_hat_Two_NN_DUDE_new[k_max*i+k-1,:]=x_nn_hat_two
           
        s_class_two=3
        s_nn_hat_cat_two=np_utils.to_categorical(s_nn_hat_two,s_class_two)
        emp_dist_two=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two=mean(sum(emp_dist_two*s_nn_hat_cat_two,axis=1))
        Est_Loss_Two_NN_DUDE_new[i,k]=est_loss_nn_dude_two
        
        ### 2-D N-DUDE Padding ###
        C_two_padding,Y_two_padding = N_DUDE.make_data_for_Two_NN_DUDE_PD(P_padding[i],P[i],k,
                                                                          L_new[i*alpha_size:(i+1)*alpha_size,],
                                                                          nb_classes,n,offset,dp)
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
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
        print '2-D N-DUDE_Padding =', error_nn_two_padding
        
        Error_Two_NN_DUDE_PD_Pre[i,k]=error_nn_two_padding
        X_hat_Two_NN_DUDE_PD_Pre[k_max*i+k-1,:]=x_nn_hat_two_padding
       
        s_class_two_padding=3
        s_nn_hat_cat_two_padding=np_utils.to_categorical(s_nn_hat_two_padding,s_class_two_padding)
        emp_dist_two_padding=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two_padding=mean(sum(emp_dist_two_padding*s_nn_hat_cat_two_padding,axis=1))
        Est_Loss_Two_NN_DUDE_PD_Pre[i,k]=est_loss_nn_dude_two_padding
        
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_PD_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_PD_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_PD_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_PD_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        tmp=model.get_weights()
        tmp[0]=saved[0]
        tmp[1]=saved[1] 
        model.set_weights(tmp) # set new weights 
       
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_padding,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_padding))
        
        # -----------------------------------------------------
    
        s_nn_hat_two_padding=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_padding=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two_padding) 
        error_nn_two_padding=N_DUDE.error_rate(x,x_nn_hat_two_padding)
        print '2-D N-DUDE_Padding =', error_nn_two_padding
        
        Error_Two_NN_DUDE_PD[i,k]=error_nn_two_padding
        X_hat_Two_NN_DUDE_PD[k_max*i+k-1,:]=x_nn_hat_two_padding
       
        s_class_two_padding=3
        s_nn_hat_cat_two_padding=np_utils.to_categorical(s_nn_hat_two_padding,s_class_two_padding)
        emp_dist_two_padding=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two_padding=mean(sum(emp_dist_two_padding*s_nn_hat_cat_two_padding,axis=1))
        Est_Loss_Two_NN_DUDE_PD[i,k]=est_loss_nn_dude_two_padding
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_PD_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_PD_Pre_trained_weights.h5",overwrite=True)
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_PD_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_PD_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal',trainable=False))
        model.add(Activation('relu',trainable=False))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        tmp=model.get_weights()
        tmp[0]=saved[0]
        tmp[1]=saved[1]
        tmp[2]=saved[2]
        tmp[3]=saved[3]
        model.set_weights(tmp) # set new weights 
       
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_padding,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_padding))
        
        # -----------------------------------------------------
    
        s_nn_hat_two_padding=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_padding=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two_padding) 
        error_nn_two_padding=N_DUDE.error_rate(x,x_nn_hat_two_padding)
        print '2-D N-DUDE_Padding =', error_nn_two_padding
        
        Error_Two_NN_DUDE_PD_new[i,k]=error_nn_two_padding
        X_hat_Two_NN_DUDE_PD_new[k_max*i+k-1,:]=x_nn_hat_two_padding
       
        s_class_two_padding=3
        s_nn_hat_cat_two_padding=np_utils.to_categorical(s_nn_hat_two_padding,s_class_two_padding)
        emp_dist_two_padding=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_two_padding=mean(sum(emp_dist_two_padding*s_nn_hat_cat_two_padding,axis=1))
        Est_Loss_Two_NN_DUDE_PD_new[i,k]=est_loss_nn_dude_two_padding
        
        Total_End=time.time()
        
        print ""
        print 'Total_Time=', Total_End-Total_Start
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Layer_wise_Two'
        np.savez(res_file,Error_Two_DUDE=Error_Two_DUDE,
                 Error_Two_NN_DUDE_Pre=Error_Two_NN_DUDE_Pre,Error_Two_NN_DUDE=Error_Two_NN_DUDE,
                 Error_Two_NN_DUDE_PD_Pre=Error_Two_NN_DUDE_PD_Pre, Error_Two_NN_DUDE_PD=Error_Two_NN_DUDE_PD,
                 Error_Two_NN_DUDE_PD_new=Error_Two_NN_DUDE_PD_new, Error_Two_NN_DUDE_new=Error_Two_NN_DUDE_new,
                 Est_Loss_Two_NN_DUDE=Est_Loss_Two_NN_DUDE,Est_Loss_Two_NN_DUDE_Pre=Est_Loss_Two_NN_DUDE_Pre,
                 Est_Loss_Two_NN_DUDE_PD=Est_Loss_Two_NN_DUDE_PD, Est_Loss_Two_NN_DUDE_PD_Pre=Est_Loss_Two_NN_DUDE_PD_Pre,
                X_hat_Two_NN_DUDE_Pre=X_hat_Two_NN_DUDE_Pre, X_hat_Two_NN_DUDE=X_hat_Two_NN_DUDE,
                  X_hat_Two_NN_DUDE_new=X_hat_Two_NN_DUDE_new, 
                 X_hat_Two_NN_DUDE_PD=X_hat_Two_NN_DUDE_PD, X_hat_Two_NN_DUDE_PD_Pre=X_hat_Two_NN_DUDE_PD_Pre,
                X_hat_Two_NN_DUDE_PD_new=X_hat_Two_NN_DUDE_PD_new)