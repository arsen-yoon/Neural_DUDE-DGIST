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
#delta = np.arange(0.005, 0.011, 0.001)
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
Error_One_NN_DUDE_Pre=Error_One_DUDE.copy()
Error_One_NN_DUDE=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD_Pre=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD=Error_One_DUDE.copy()
Error_One_NN_DUDE_LB_Pre=Error_One_DUDE.copy()
Error_One_NN_DUDE_LB=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD_LB_Pre=Error_One_DUDE.copy()
Error_One_NN_DUDE_PD_LB=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind_Pre=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind_Norm_Pre=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind_Norm=Error_One_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_One_DUDE[i,0]=delta[i]
    Error_One_NN_DUDE_Pre[i,0]=delta[i]
    Error_One_NN_DUDE[i,0]=delta[i]
    Error_One_NN_DUDE_PD_Pre[i,0]=delta[i]
    Error_One_NN_DUDE_PD[i,0]=delta[i]
    Error_One_NN_DUDE_LB_Pre[i,0]=delta[i]
    Error_One_NN_DUDE_LB[i,0]=delta[i]
    Error_One_NN_DUDE_PD_LB_Pre[i,0]=delta[i]
    Error_One_NN_DUDE_PD_LB[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Pre[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Norm_Pre[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Norm[i,0]=delta[i]
    
### Est Loss Mat ###
Est_Loss_One_NN_DUDE_Pre=zeros((len(delta),k_max+1))
Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_PD_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_PD=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_LB_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_LB=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_PD_LB_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
Est_Loss_One_NN_DUDE_PD_LB=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_CB_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_CB=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_Norm_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_Norm=Est_Loss_One_NN_DUDE_Pre.copy()

for i in range(len(delta)):
    Est_Loss_One_NN_DUDE_Pre[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE[i,0]=delta[i]
    Est_Loss_One_NN_DUDE_PD_Pre[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE_PD[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE_LB_Pre[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE_LB[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE_PD_LB_Pre[i,0]=delta[i] 
    Est_Loss_One_NN_DUDE_PD_LB[i,0]=delta[i] 
    #Est_Loss_One_NN_DUDE_CB_Pre[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_CB[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_Norm_Pre[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_Norm[i,0]=delta[i]
    
### X_hat Mat ###
X_hat_One_DUDE=zeros((len(delta)*k_max,n))
X_hat_One_NN_DUDE_Pre=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD_Pre=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_LB_Pre=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_LB=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD_LB_Pre=X_hat_One_DUDE.copy()
X_hat_One_NN_DUDE_PD_LB=X_hat_One_DUDE.copy()

#X_hat_One_NN_DUDE_Bind_Pre=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind_Norm_Pre=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind_Norm=X_hat_One_DUDE.copy()
'''
lr 낮춰서 0.01, 효과 있으면 기존의 다른 방법에도 다시 적용. layer by layer decay? or time decay? or 'k' decay?
'''
for i in range(0,1):
    #print "##### delta=%0.2f #####" % delta[i]
    for k in range(1, k_max+1):
        print "k =",k
        One_NN_Start=time.time()
        
        ### 1-D DUDE ###
        s_hat,m=DUDE.One_DUDE(z[0],k,delta[0])
        x_dude_hat=DUDE.denoise_with_s(z[0],s_hat,k)
        error_dude=DUDE.error_rate(x,x_dude_hat)
        print '1-D DUDE =',error_dude
        Error_One_DUDE[0,k]=error_dude
        X_hat_One_DUDE[k_max*0+k-1,:]=x_dude_hat
        
        ### 1-D N-DUDE ###
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
        model.fit(C,Y,nb_epoch=10,batch_size=128,show_accuracy=False, verbose=0)
        
        pred_class=model.predict_classes(C, batch_size=128, verbose=0)
        s_nn_hat=hstack((zeros(k),pred_class,zeros(k)))
        x_nn_hat=N_DUDE.denoise_with_s(z[i],s_nn_hat,k)
        error_nn=N_DUDE.error_rate(x,x_nn_hat)
        
        print '1-D N-DUDE Pre-trained =', error_nn
         
        Error_One_NN_DUDE_Pre[i,k]=error_nn
        X_hat_One_NN_DUDE_Pre[k_max*i,:]=x_nn_hat
        
        s_class=3
        s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))
        Est_Loss_One_NN_DUDE_Pre[i,k]=est_loss_nn_dude
        
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("NN_DUDE_Pre_trained_Model_ver3_4.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("NN_DUDE_Pre_trained_weights_ver3_4.h5",overwrite=True)
        
        # -----------------------------------------------------
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("NN_DUDE_Pre_trained_Model_ver3_4.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("NN_DUDE_Pre_trained_weights_ver3_4.h5")
        saved=loaded_model.get_weights()
        
        C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[0*n:(1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        
        ### 1-D N-DUDE ###
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.set_weights(saved) # set new weights 
       
        model.compile(loss='poisson', optimizer=adam)
        model.optimizer.lr.set_value(0.003)
        logs.get('loss'
        model.fit(C,Y,nb_epoch=10,batch_size=128,show_accuracy=False, verbose=0)
        
        pred_class=model.predict_classes(C, batch_size=128, verbose=0)
        s_nn_hat=hstack((zeros(k),pred_class,zeros(k)))
        x_nn_hat=N_DUDE.denoise_with_s(z[0],s_nn_hat,k)
        error_nn=N_DUDE.error_rate(x,x_nn_hat)
        
        print '1-D N-DUDE After =', error_nn
    
        Error_One_NN_DUDE[0,k]=error_nn
        X_hat_One_NN_DUDE[k_max*0,:]=x_nn_hat
        
        s_class=3
        s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)
        emp_dist=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))
        Est_Loss_One_NN_DUDE[0,k]=est_loss_nn_dude
        '''
        ### 1-D N-DUDE Bound ###
        L_lower=np.array([[1-delta[i],1,0],[1-delta[i],1,0],[1-delta[i],0,1],[1-delta[i],0,1]])
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
    
        print '1-D N-DUDE_Bound Pre-trained =', error_nn_lower
        Error_One_NN_DUDE_LB_Pre[i,k]=error_nn_lower
    
        X_hat_One_NN_DUDE_LB_Pre[k_max*i+k-1,:]=x_nn_hat_lower
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("NN_DUDE_LB_Pre_trained_Model_ver3_3.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("NN_DUDE_LB_Pre_trained_weights_ver3_3.h5",overwrite=True)
        
        # -----------------------------------------------------
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("NN_DUDE_LB_Pre_trained_Model_ver3_3.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("NN_DUDE_LB_Pre_trained_weights_ver3_3.h5")
        saved=loaded_model.get_weights()
        
        ### 1-D N-DUDE Bound ###
        L_lower=np.array([[1-delta[0],1,0],[1-delta[0],1,0],[1-delta[0],0,1],[1-delta[0],0,1]])
        Y_lower = N_DUDE.make_data_for_One_NN_DUDE_LB(Z[0*n:(0+1)*n],k,L_lower,x,z[0],nb_classes,n)
    
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        #model.fit(C,Y_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C, Y_lower))
    
        # -----------------------------------------------------
    
        pred_class_lower=model.predict_classes(C, batch_size=200, verbose=0)
        s_nn_hat_lower=hstack((zeros(k),pred_class_lower,zeros(k)))
        x_nn_hat_lower=N_DUDE.denoise_with_s(z[0],s_nn_hat_lower,k)
        error_nn_lower=N_DUDE.error_rate(x,x_nn_hat_lower)
    
        print '1-D N-DUDE_Bound =', error_nn_lower
        Error_One_NN_DUDE_LB[0,k]=error_nn_lower
    
        X_hat_One_NN_DUDE_LB[k_max*0+k-1,:]=x_nn_hat_lower
        
        
        # -------------------------------------------------------- #
        ### 1-D N-DUDE Padding ###
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
        
        s_nn_hat_PD=model.predict_classes(C_PD, batch_size=200, verbose=0)
        x_nn_hat_PD=N_DUDE.denoise_with_s_One_NN_PD(z[i],s_nn_hat_PD)
        error_nn_PD=N_DUDE.error_rate(x,x_nn_hat_PD)
        
        print '1-D N-DUDE Padding Pre-trained =', error_nn_PD
        
        Error_One_NN_DUDE_PD_Pre[i,k]=error_nn_PD
        X_hat_One_NN_DUDE_PD_Pre[k_max*i+k-1,:]=x_nn_hat_PD
        
        s_class=3
        s_nn_hat_cat_PD=np_utils.to_categorical(s_nn_hat_PD,s_class)
        emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        est_loss_nn_dude_PD=mean(sum(emp_dist*s_nn_hat_cat_PD,axis=1))
        Est_Loss_One_NN_DUDE_PD_Pre[i,k]=est_loss_nn_dude_PD
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("NN_DUDE_PD_Pre_trained_Model_ver3_3.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("NN_DUDE_PD_Pre_trained_weights_ver3_3.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("NN_DUDE_PD_Pre_trained_Model_ver3_3.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("NN_DUDE_PD_Pre_trained_weights_ver3_3.h5")
        saved=loaded_model.get_weights()
        
        C_PD,Y_PD = N_DUDE.make_data_for_One_NN_DUDE_PD(P[0],Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.set_weights(saved) # set new weights 
        model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_PD,Y_PD,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_PD, Y_PD))
        
        s_nn_hat_PD=model.predict_classes(C_PD, batch_size=200, verbose=0)
        x_nn_hat_PD=N_DUDE.denoise_with_s_One_NN_PD(z[0],s_nn_hat_PD)
        error_nn_PD=N_DUDE.error_rate(x,x_nn_hat_PD)
        
        print '1-D N-DUDE Padding =', error_nn_PD
        
        Error_One_NN_DUDE_PD[0,k]=error_nn_PD
        X_hat_One_NN_DUDE_PD[k_max*0+k-1,:]=x_nn_hat_PD
        
        s_class=3
        s_nn_hat_cat_PD=np_utils.to_categorical(s_nn_hat_PD,s_class)
        emp_dist=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        est_loss_nn_dude_PD=mean(sum(emp_dist*s_nn_hat_cat_PD,axis=1))
        Est_Loss_One_NN_DUDE_PD[0,k]=est_loss_nn_dude_PD
        
        ### 1-D N-DUDE Padding Bound ###
        C_PD,Y_PD = N_DUDE.make_data_for_One_NN_DUDE_PD(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
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
    
        print '1-D N-DUDE Padding Bound Pre-trained =', error_nn_PD_lower
        Error_One_NN_DUDE_PD_LB_Pre[i,k]=error_nn_PD_lower
    
        X_hat_One_NN_DUDE_PD_LB_Pre[k_max*i+k-1,:]=x_nn_hat_PD_lower
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("NN_DUDE_PD_LB_Pre_trained_Model_ver3_3.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("NN_DUDE_PD_LB_Pre_trained_weights_ver3_3.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("NN_DUDE_PD_LB_Pre_trained_Model_ver3_3.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("NN_DUDE_PD_LB_Pre_trained_weights_ver3_3.h5")
        saved=loaded_model.get_weights()
        
        ### 1-D N-DUDE Padding Bound ###
        C_PD,Y_PD = N_DUDE.make_data_for_One_NN_DUDE_PD(P[0],Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        Y_PD_lower = N_DUDE.make_data_for_One_NN_DUDE_PD_LB(P[0],Z[0*n:(0+1)*n],im_bin,k,L_lower,nb_classes,n)
    
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal',trainable=False))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_PD,Y_PD_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_PD, Y_PD_lower))
    
        # -----------------------------------------------------
    
        s_nn_hat_PD_lower=model.predict_classes(C_PD, batch_size=200, verbose=0)
        x_nn_hat_PD_lower=N_DUDE.denoise_with_s_One_NN_PD(z[0],s_nn_hat_PD_lower)
        error_nn_PD_lower=N_DUDE.error_rate(x,x_nn_hat_PD_lower)
    
        print '1-D N-DUDE Padding Bound =', error_nn_PD_lower
        Error_One_NN_DUDE_PD_LB[0,k]=error_nn_PD_lower
    
        X_hat_One_NN_DUDE_PD_LB[k_max*0+k-1,:]=x_nn_hat_PD_lower
        
        
        # -------------------------------------------------------- #
        ### 1-D Context Bind ###
        #C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        #C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
            
        #model=Sequential()
        #model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(3,init='he_normal'))
        #model.add(Activation('softmax'))
        
        #rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        #adagrad=Adagrad(clipnorm=1.5)
        #adam=Adam()
        #adadelta=Adadelta()
        #sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        #model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_Bind,Y_Bind,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind))
        # -----------------------------------------------------
        
        #pred_class_Bind=model.predict_classes(C,batch_size=200,verbose=0)
        #s_nn_hat_Bind=hstack((zeros(k),pred_class_Bind,zeros(k)))
        #x_nn_hat_Bind=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind,k)
        #error_nn_Bind=N_DUDE.error_rate(x,x_nn_hat_Bind)
        
        #print '1-D N-DUDE Context Bind Pre-trained =', error_nn_Bind
        
        #Error_One_NN_DUDE_Bind_Pre[i,k]=error_nn_Bind
        #X_hat_One_NN_DUDE_Bind_Pre[i*k_max+k-1]=x_nn_hat_Bind
        
        #s_class=3
        #s_nn_hat_cat_Bind=np_utils.to_categorical(s_nn_hat_Bind,s_class)
        #emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        #est_loss_nn_dude_Bind=mean(sum(emp_dist*s_nn_hat_cat_Bind,axis=1))
        #Est_Loss_One_NN_DUDE_CB_Pre[i,k]=est_loss_nn_dude_Bind
        
        ### Save the model & weights ###
        #model_json=model.to_json()
        #with open("NN_DUDE_CB_Pre_trained_Model.json","w") as json_file:
        #    json_file.write(model_json)
        #model.save_weights("NN_DUDE_CB_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        #json_file=open("NN_DUDE_CB_Pre_trained_Model.json",'r')
        #loaded_model_json=json_file.read()
        #json_file.close()
        #loaded_model=model_from_json(loaded_model_json)
        #loaded_model.load_weights("NN_DUDE_CB_Pre_trained_weights.h5")
        #saved=loaded_model.get_weights()
        
        #C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        #C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[0],Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
            
        #model=Sequential()
        #model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(3,init='he_normal'))
        #model.add(Activation('softmax'))
        #model.set_weights(saved) # set new weights 
        
        #rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        #adagrad=Adagrad(clipnorm=1.5)
        #adam=Adam()
        #adadelta=Adadelta()
        #sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)
        
        #model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_Bind,Y_Bind,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind))
        
        #pred_class_Bind=model.predict_classes(C,batch_size=200,verbose=0)
        #s_nn_hat_Bind=hstack((zeros(k),pred_class_Bind,zeros(k)))
        #x_nn_hat_Bind=N_DUDE.denoise_with_s(z[0],s_nn_hat_Bind,k)
        #error_nn_Bind=N_DUDE.error_rate(x,x_nn_hat_Bind)
        
        #print '1-D N-DUDE Context Bind =', error_nn_Bind
        
        #Error_One_NN_DUDE_Bind[0,k]=error_nn_Bind
        #X_hat_One_NN_DUDE_Bind[0*k_max+k-1]=x_nn_hat_Bind
        
        #s_class=3
        #s_nn_hat_cat_Bind=np_utils.to_categorical(s_nn_hat_Bind,s_class)
        #emp_dist=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        #est_loss_nn_dude_Bind=mean(sum(emp_dist*s_nn_hat_cat_Bind,axis=1))
        #Est_Loss_One_NN_DUDE_CB[0,k]=est_loss_nn_dude_Bind
        
        # -------------------------------------------------------- #
        ### 1-D Context Bind Norm ###
        #C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        #C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
        #model=Sequential()
        #model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(3,init='he_normal'))
        #model.add(Activation('softmax'))
        #model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_Bind,Y_Bind_Norm,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind_Norm))
        
        #pred_class_Bind_Norm=model.predict_classes(C,batch_size=200,verbose=0)
        #s_nn_hat_Bind_Norm=hstack((zeros(k),pred_class_Bind_Norm,zeros(k)))
        #x_nn_hat_Bind_Norm=N_DUDE.denoise_with_s(z[i],s_nn_hat_Bind_Norm,k)
        #error_nn_Bind_Norm=N_DUDE.error_rate(x,x_nn_hat_Bind_Norm)
        
        #print '1-D N-DUDE Context Bind Normalization Pre-trained =', error_nn_Bind_Norm
        
        #Error_One_NN_DUDE_Bind_Norm_Pre[i,k]=error_nn_Bind_Norm
        #X_hat_One_NN_DUDE_Bind_Norm_Pre[i*k_max+k-1]=x_nn_hat_Bind_Norm
        
        #s_class=3
        #s_nn_hat_cat_Bind_Norm=np_utils.to_categorical(s_nn_hat_Bind_Norm,s_class)
        #emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
        #est_loss_nn_dude_Bind_Norm=mean(sum(emp_dist*s_nn_hat_cat_Bind_Norm,axis=1))
        #Est_Loss_One_NN_DUDE_Norm_Pre[i,k]=est_loss_nn_dude_Bind_Norm
        
        ### Save the model & weights ###
        #model_json=model.to_json()
        #with open("NN_DUDE_CB_Norm_Pre_trained_Model.json","w") as json_file:
        #    json_file.write(model_json)
        #model.save_weights("NN_DUDE_CB_Norm_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        #json_file=open("NN_DUDE_CB_Norm_Pre_trained_Model.json",'r')
        #loaded_model_json=json_file.read()
        #json_file.close()
        #loaded_model=model_from_json(loaded_model_json)
        #loaded_model.load_weights("NN_DUDE_CB_Norm_Pre_trained_weights.h5")
        #saved=loaded_model.get_weights()
        
        #C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        #C_Bind,Y_Bind,Y_Bind_Norm,key,m=N_DUDE.make_data_for_One_NN_DUDE_Context_Bind(z[0],Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n)
        
        #model=Sequential()
        #model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(40,init='he_normal'))
        #model.add(Activation('relu'))
        #model.add(Dense(3,init='he_normal'))
        #model.add(Activation('softmax'))
        #model.compile(loss='poisson', optimizer=adam)
        #model.fit(C_Bind,Y_Bind_Norm,nb_epoch=10,show_accuracy=True, verbose=0, validation_data=(C_Bind, Y_Bind_Norm))
        
        #pred_class_Bind_Norm=model.predict_classes(C,batch_size=200,verbose=0)
        #s_nn_hat_Bind_Norm=hstack((zeros(k),pred_class_Bind_Norm,zeros(k)))
        #x_nn_hat_Bind_Norm=N_DUDE.denoise_with_s(z[0],s_nn_hat_Bind_Norm,k)
        #error_nn_Bind_Norm=N_DUDE.error_rate(x,x_nn_hat_Bind_Norm)
        
        #print '1-D N-DUDE Context Bind Normalization =', error_nn_Bind_Norm
        
        #Error_One_NN_DUDE_Bind_Norm[0,k]=error_nn_Bind_Norm
        #X_hat_One_NN_DUDE_Bind_Norm[0*k_max+k-1]=x_nn_hat_Bind_Norm
        
        #s_class=3
        #s_nn_hat_cat_Bind_Norm=np_utils.to_categorical(s_nn_hat_Bind_Norm,s_class)
        #emp_dist=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        #est_loss_nn_dude_Bind_Norm=mean(sum(emp_dist*s_nn_hat_cat_Bind_Norm,axis=1))
        #Est_Loss_One_NN_DUDE_Norm[0,k]=est_loss_nn_dude_Bind_Norm
        
        '''
        One_NN_End=time.time()
        One_NN_Duration=One_NN_End-One_NN_Start
        
        print 'Time =', One_NN_Duration
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Fine_Tune_One_new_ver2'
        np.savez(res_file, Error_One_DUDE=Error_One_DUDE, Error_One_NN_DUDE=Error_One_NN_DUDE,
                 Error_One_NN_DUDE_LB_Pre=Error_One_NN_DUDE_LB_Pre, Error_One_NN_DUDE_LB=Error_One_NN_DUDE_LB,
                 Error_One_NN_DUDE_PD_LB_Pre=Error_One_NN_DUDE_PD_LB_Pre,Error_One_NN_DUDE_PD_LB=Error_One_NN_DUDE_PD_LB,
                 Error_One_NN_DUDE_PD=Error_One_NN_DUDE_PD, Error_One_NN_DUDE_PD_Pre=Error_One_NN_DUDE_PD_Pre,
                 Error_One_NN_DUDE_Pre=Error_One_NN_DUDE_Pre,
                 X_hat_One_NN_DUDE=X_hat_One_NN_DUDE, X_hat_One_NN_DUDE_Pre=X_hat_One_NN_DUDE_Pre, Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE, Est_Loss_One_NN_DUDE_Pre=Est_Loss_One_NN_DUDE_Pre, Est_Loss_One_NN_DUDE_PD=Est_Loss_One_NN_DUDE_PD, Est_Loss_One_NN_DUDE_PD_Pre=Est_Loss_One_NN_DUDE_PD_Pre)