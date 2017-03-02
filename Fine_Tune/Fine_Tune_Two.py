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
Error_Two_DUDE=zeros((len(delta),k_max+1))
Error_Two_NN_DUDE_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_LB_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_LB=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD_LB_Pre=Error_Two_DUDE.copy()
Error_Two_NN_DUDE_PD_LB=Error_Two_DUDE.copy()
#Error_One_NN_DUDE_Bind_Pre=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind_Norm_Pre=Error_One_DUDE.copy()
#Error_One_NN_DUDE_Bind_Norm=Error_One_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_Two_DUDE[i,0]=delta[i]
    Error_Two_NN_DUDE_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE_PD[i,0]=delta[i]
    Error_Two_NN_DUDE_LB_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE_LB[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_LB_Pre[i,0]=delta[i]
    Error_Two_NN_DUDE_PD_LB[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Pre[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Norm_Pre[i,0]=delta[i]
    #Error_One_NN_DUDE_Bind_Norm[i,0]=delta[i]
    
### Est Loss Mat ###
Est_Loss_Two_NN_DUDE_Pre=zeros((len(delta),k_max+1))
Est_Loss_Two_NN_DUDE=Est_Loss_Two_NN_DUDE_Pre.copy()
Est_Loss_Two_NN_DUDE_PD_Pre=Est_Loss_Two_NN_DUDE_Pre.copy()
Est_Loss_Two_NN_DUDE_PD=Est_Loss_Two_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_CB_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_CB=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_Norm_Pre=Est_Loss_One_NN_DUDE_Pre.copy()
#Est_Loss_One_NN_DUDE_Norm=Est_Loss_One_NN_DUDE_Pre.copy()

for i in range(len(delta)):
    Est_Loss_Two_NN_DUDE_Pre[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE[i,0]=delta[i]
    Est_Loss_Two_NN_DUDE_PD_Pre[i,0]=delta[i] 
    Est_Loss_Two_NN_DUDE_PD[i,0]=delta[i] 
    #Est_Loss_One_NN_DUDE_CB_Pre[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_CB[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_Norm_Pre[i,0]=delta[i]
    #Est_Loss_One_NN_DUDE_Norm[i,0]=delta[i]
    
### X_hat Mat ###
X_hat_Two_DUDE=zeros((len(delta)*k_max,n))
X_hat_Two_NN_DUDE_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_LB_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_LB=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD_LB_Pre=X_hat_Two_DUDE.copy()
X_hat_Two_NN_DUDE_PD_LB=X_hat_Two_DUDE.copy()

#X_hat_One_NN_DUDE_Bind_Pre=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind_Norm_Pre=X_hat_One_DUDE.copy()
#X_hat_One_NN_DUDE_Bind_Norm=X_hat_One_DUDE.copy()

for i in range(4,5):
    #print "##### delta=%0.2f #####" % delta[i]
    for k in range(1, k_max+1):
        print "k =",k
        Two_NN_Start=time.time()
        
        ### 2-D DUDE ###
        s_hat_two,m=DUDE.Two_DUDE(z_two[0],k,delta[0],n,offset) 
        x_dude_hat_two=DUDE.denoise_with_s_Two_DUDE(z_two[0],s_hat_two,k)
        error_dude_two=DUDE.error_rate(x,x_dude_hat_two)
        print '2-D DUDE =',error_dude_two
        
        Error_Two_DUDE[0,k]=error_dude_two
        X_hat_Two_DUDE[k_max*0+k-1,:]=x_dude_hat_two
        
        ### 2-D N-DUDE ###
        C_two,Y_two = N_DUDE.make_data_for_Two_NN_DUDE(P[i],Z[i*n:(i+1)*n],k,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n,offset)
        
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
        
        # -----------------------------------------------------
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        C_two,Y_two = N_DUDE.make_data_for_Two_NN_DUDE(P[0],Z[0*n:(0+1)*n],k,L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n,offset)
        
        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.set_weights(saved) # set new weights 
        
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two, Y_two))
        
        # -----------------------------------------------------
    
        pred_class_two=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[0],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE =', error_nn_two
        
        Error_Two_NN_DUDE[0,k]=error_nn_two
        X_hat_Two_NN_DUDE[k_max*0+k-1,:]=x_nn_hat_two
           
        s_class_two=3
        s_nn_hat_cat_two=np_utils.to_categorical(s_nn_hat_two,s_class_two)
        emp_dist_two=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        est_loss_nn_dude_two=mean(sum(emp_dist_two*s_nn_hat_cat_two,axis=1))
        Est_Loss_Two_NN_DUDE[0,k]=est_loss_nn_dude_two
        
        ### 2-D N-DUDE Bound ###
        L_lower=np.array([[1-delta[i],1,0],[1-delta[i],1,0],[1-delta[i],0,1],[1-delta[i],0,1]])
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
    
        print '2-D N-DUDE_Bound Pre-trained =', error_nn_two_lower
        Error_Two_NN_DUDE_LB_Pre[i,k]=error_nn_two_lower
        X_hat_Two_NN_DUDE_LB_Pre[k_max*i+k-1,:]=x_nn_hat_two_lower
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_LB_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_LB_Pre_trained_weights.h5",overwrite=True)
        
        # -----------------------------------------------------
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_LB_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_LB_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        ### 2-D N-DUDE Bound ###
        L_lower=np.array([[1-delta[0],1,0],[1-delta[0],1,0],[1-delta[i],0,1],[1-delta[0],0,1]])
        Y_two_lower = N_DUDE.make_data_for_Two_NN_DUDE_LB(P[0],k,L_lower,im_bin,nb_classes,n,offset)

        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.set_weights(saved) # set new weights 
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two,Y_two_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two, Y_two_lower))
    
        # -----------------------------------------------------
    
        pred_class_two_lower=model.predict_classes(C_two, batch_size=200, verbose=0)
        s_nn_hat_two_lower=N_DUDE.mapping_mat_resize(pred_class_two_lower,k,n)
        x_nn_hat_two_lower=N_DUDE.denoise_with_s_Two_NN_DUDE(z[0],s_nn_hat_two_lower)
        error_nn_two_lower=N_DUDE.error_rate(x,x_nn_hat_two_lower)
    
        print '2-D N-DUDE_Bound =', error_nn_two_lower
        Error_Two_NN_DUDE_LB[0,k]=error_nn_two_lower
        X_hat_Two_NN_DUDE_LB[k_max*0+k-1,:]=x_nn_hat_two_lower
        
        # -------------------------------------------------------- #
        ### 2-D N-DUDE Padding ###
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
        
        ### 2-D N-DUDE Padding ###
        C_two_padding,Y_two_padding = N_DUDE.make_data_for_Two_NN_DUDE_PD(P_padding[0],P[0],k,
                                                                          L_new[0*alpha_size:(0+1)*alpha_size,],
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
        model.set_weights(saved) # set new weights 
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_padding,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_padding))
        
        # -----------------------------------------------------
    
        s_nn_hat_two_padding=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_padding=N_DUDE.denoise_with_s_Two_NN_DUDE(z[0],s_nn_hat_two_padding) 
        error_nn_two_padding=N_DUDE.error_rate(x,x_nn_hat_two_padding)
        print '2-D N-DUDE_Padding =', error_nn_two_padding
        
        Error_Two_NN_DUDE_PD[0,k]=error_nn_two_padding
        X_hat_Two_NN_DUDE_PD[k_max*0+k-1,:]=x_nn_hat_two_padding
       
        s_class_two_padding=3
        s_nn_hat_cat_two_padding=np_utils.to_categorical(s_nn_hat_two_padding,s_class_two_padding)
        emp_dist_two_padding=dot(Z[0*n:(0+1)*n,],L[0*alpha_size:(0+1)*alpha_size,])
        est_loss_nn_dude_two_padding=mean(sum(emp_dist_two_padding*s_nn_hat_cat_two_padding,axis=1))
        Est_Loss_Two_NN_DUDE_PD[0,k]=est_loss_nn_dude_two_padding
        
        ### 2-D N-DUDE Padding Bound ###
        L_lower=np.array([[1-delta[i],1,0],[1-delta[i],1,0],[1-delta[i],0,1],[1-delta[i],0,1]])
        C_two_padding,Y_two_padding = N_DUDE.make_data_for_Two_NN_DUDE_PD(P_padding[i],P[i],k,
                                                                          L_new[i*alpha_size:(i+1)*alpha_size,],
                                                                          nb_classes,n,offset,dp)
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
    
        print '2-D N-DUDE_PD_Bound Pre-trained =', error_nn_two_PD_lower
        Error_Two_NN_DUDE_PD_LB_Pre[i,k]=error_nn_two_PD_lower
        X_hat_Two_NN_DUDE_PD_LB_Pre[k_max*i+k-1,:]=x_nn_hat_two_PD_lower
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("Two_NN_DUDE_PD_LB_Pre_trained_Model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("Two_NN_DUDE_PD_LB_Pre_trained_weights.h5",overwrite=True)
        
        # -------------------------------------------------------#
        # Load pre-trained model and feed low delta data #
        
        ### Load the model & weights ###
        json_file=open("Two_NN_DUDE_PD_LB_Pre_trained_Model.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("Two_NN_DUDE_PD_LB_Pre_trained_weights.h5")
        saved=loaded_model.get_weights()
        
        ### 2-D N-DUDE Padding Bound ###
        L_lower=np.array([[1-delta[0],1,0],[1-delta[0],1,0],[1-delta[0],0,1],[1-delta[0],0,1]])
        C_two_padding,Y_two_padding = N_DUDE.make_data_for_Two_NN_DUDE_PD(P_padding[0],P[0],k,
                                                                          L_new[0*alpha_size:(0+1)*alpha_size,],
                                                                          nb_classes,n,offset,dp)
        Y_two_PD_lower = N_DUDE.make_data_for_Two_NN_DUDE_PD_LB(P_padding[0],P[0],im_bin,k,L_lower,nb_classes,n,offset,dp)

        model=Sequential()
        model.add(Dense(40,input_dim=2*k*nb_classes,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(40,init='he_normal'))
        model.add(Activation('relu'))
        model.add(Dense(3,init='he_normal'))
        model.add(Activation('softmax'))
        model.set_weights(saved) # set new weights 
        model.compile(loss='poisson', optimizer=adam)
        model.fit(C_two_padding,Y_two_PD_lower,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, 
                  validation_data=(C_two_padding, Y_two_PD_lower))
    
        # -----------------------------------------------------
    
        s_nn_hat_two_PD_lower=model.predict_classes(C_two_padding, batch_size=200, verbose=0)
        x_nn_hat_two_PD_lower=N_DUDE.denoise_with_s_Two_NN_DUDE(z[0],s_nn_hat_two_PD_lower)
        error_nn_two_PD_lower=N_DUDE.error_rate(x,x_nn_hat_two_PD_lower)
    
        print '2-D N-DUDE_PD_Bound =', error_nn_two_PD_lower
        Error_Two_NN_DUDE_PD_LB[0,k]=error_nn_two_PD_lower
        X_hat_Two_NN_DUDE_PD_LB[k_max*0+k-1,:]=x_nn_hat_two_PD_lower
        
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
        
        Two_NN_End=time.time()
        Two_NN_Duration=Two_NN_End-Two_NN_Start
        
        print 'Time =', Two_NN_Duration
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Fine_Tune_Two'
        np.savez(res_file, Error_Two_DUDE=Error_Two_DUDE, Error_Two_NN_DUDE=Error_Two_NN_DUDE,
                 Error_Two_NN_DUDE_LB_Pre=Error_Two_NN_DUDE_LB_Pre, Error_Two_NN_DUDE_LB=Error_Two_NN_DUDE_LB,
                 Error_Two_NN_DUDE_PD_LB_Pre=Error_Two_NN_DUDE_PD_LB_Pre,Error_Two_NN_DUDE_PD_LB=Error_Two_NN_DUDE_PD_LB,
                 Error_Two_NN_DUDE_PD=Error_Two_NN_DUDE_PD, Error_Two_NN_DUDE_PD_Pre=Error_Two_NN_DUDE_PD_Pre,
                 Error_Two_NN_DUDE_Pre=Error_Two_NN_DUDE_Pre,
                 X_hat_Two_NN_DUDE=X_hat_Two_NN_DUDE, X_hat_Two_NN_DUDE_Pre=X_hat_Two_NN_DUDE_Pre, Est_Loss_Two_NN_DUDE=Est_Loss_Two_NN_DUDE, Est_Loss_Two_NN_DUDE_Pre=Est_Loss_Two_NN_DUDE_Pre, Est_Loss_Two_NN_DUDE_PD=Est_Loss_Two_NN_DUDE_PD, Est_Loss_Two_NN_DUDE_PD_Pre=Est_Loss_Two_NN_DUDE_PD_Pre)