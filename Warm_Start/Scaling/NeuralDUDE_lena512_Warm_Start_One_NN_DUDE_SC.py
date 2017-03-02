import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN, device=cpu,floatX=float32"
import theano
import keras
import time

import numpy as np
import Binary_DUDE as DUDE
import Binary_N_DUDE as N_DUDE
import Warm_Start as WS

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
    print 'k=1'
    C,Y = N_DUDE.make_data_for_One_NN_DUDE(Z[i*n:(i+1)*n],1,L_new[i*alpha_size:(i+1)*alpha_size,],nb_classes,n)
        
    model=Sequential()
    model.add(Dense(40,input_dim=2*1*nb_classes,init='he_normal'))
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
    s_nn_hat=hstack((zeros(1),pred_class,zeros(1)))
    x_nn_hat=N_DUDE.denoise_with_s(z[i],s_nn_hat,1)
    error_nn=N_DUDE.error_rate(x,x_nn_hat)
        
    print '1-D N-DUDE=', error_nn
        
    Error_One_NN_DUDE[i,1]=error_nn
    X_hat_One_NN_DUDE[k_max*i,:]=x_nn_hat
        
    s_class=3
    s_nn_hat_cat=np_utils.to_categorical(s_nn_hat,s_class)
    emp_dist=dot(Z[i*n:(i+1)*n,],L[i*alpha_size:(i+1)*alpha_size,])
    est_loss_nn_dude=mean(sum(emp_dist*s_nn_hat_cat,axis=1))
    Est_Loss_One_NN_DUDE[i,1]=est_loss_nn_dude
    
    ### Save the model & weights ###
    model_json=model.to_json()
    with open("saved_model_NN_DUDE.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("saved_weights_NN_DUDE.h5",overwrite=True)
    
    for k in range(2,k_max+1):
        print 'k=',k
        ### Load the model & weights ###
        json_file=open("saved_model_NN_DUDE.json",'r')
        loaded_model_json=json_file.read()
        json_file.close()
        loaded_model=model_from_json(loaded_model_json)
        loaded_model.load_weights("saved_weights_NN_DUDE.h5")
        
        ### Add randomly initialized weights ###
        before_dim=loaded_model.get_weights()[0].shape[0]
        num_nodes=loaded_model.get_weights()[0].shape[1] # num of nodes in the layer
        # For every increasing window size, 4 dimension increase because of one-hot encoding
        new_rand=WS.rand_initialization(4,40)
        new_weights=np.vstack((new_rand[0:2], multiply(loaded_model.get_weights()[0], (k-1)/float(k)), new_rand[1:3]))
        saved=loaded_model.get_weights()
        saved[0]=new_weights
        
        ### 1-D N-DUDE ###
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
        model.set_weights(saved) # set new weights 
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
        
        ### Save the model & weights ###
        model_json=model.to_json()
        with open("saved_model_NN_DUDE.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights("saved_weights_NN_DUDE.h5",overwrite=True)
    
        One_NN_End=time.time()
        One_NN_Duration=One_NN_End-One_NN_Start
        
        print ""
        print '1-D_NN_Time=', One_NN_Duration
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Warm_Start_One_NN_DUDE_Scaling'
        np.savez(res_file,Error_One_NN_DUDE=Error_One_NN_DUDE,
                 X_hat_One_NN_DUDE=X_hat_One_NN_DUDE,Est_Loss_One_NN_DUDE=Est_Loss_One_NN_DUDE)