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
Error_Two_NN_DUDE=Error_Two_DUDE.copy()

### Mat initialization ###
for i in range(len(delta)):
    Error_Two_DUDE[i,0]=delta[i]
    Error_Two_NN_DUDE[i,0]=delta[i]
    
### Est Loss Mat ###
Est_Loss_Two_NN_DUDE=zeros((len(delta),k_max+1))

for i in range(len(delta)):
    Est_Loss_Two_NN_DUDE[i,0]=delta[i]
    
### X_hat Mat ###
X_hat_Two_DUDE=zeros((len(delta)*k_max,n))
X_hat_Two_NN_DUDE=X_hat_Two_DUDE.copy()

for i in range(0,1):
    #print "##### delta=%0.2f #####" % delta[i]
    for k in range(1, k_max+1):
        print "k =",k
        One_NN_Start=time.time()
        
        ### 2-D DUDE ###
        Two_DUDE_Start=time.time()
        s_hat_two,m=DUDE.Two_DUDE(z_two[i],k,delta[i],n,offset) 
        x_dude_hat_two=DUDE.denoise_with_s_Two_DUDE(z_two[i],s_hat_two,k)
        error_dude_two=DUDE.error_rate(x,x_dude_hat_two)
        print '2-D DUDE =',error_dude_two
        
        Error_Two_DUDE[i,k]=error_dude_two
        X_hat_Two_DUDE[k_max*i+k-1,:]=x_dude_hat_two
        
        ### 2-D N-DUDE ###
        C_two1,Y_two1 = N_DUDE.make_data_for_Two_NN_DUDE(P[0],Z[0*n:(0+1)*n],k,
                                                       L_new[0*alpha_size:(0+1)*alpha_size,],nb_classes,n,offset)
        C_two2,Y_two2 = N_DUDE.make_data_for_Two_NN_DUDE(P[1],Z[1*n:(1+1)*n],k,
                                                       L_new[1*alpha_size:(1+1)*alpha_size,],nb_classes,n,offset)
        C_two3,Y_two3 = N_DUDE.make_data_for_Two_NN_DUDE(P[2],Z[2*n:(2+1)*n],k,
                                                       L_new[2*alpha_size:(2+1)*alpha_size,],nb_classes,n,offset)
        C_two4,Y_two4 = N_DUDE.make_data_for_Two_NN_DUDE(P[3],Z[3*n:(3+1)*n],k,
                                                       L_new[3*alpha_size:(3+1)*alpha_size,],nb_classes,n,offset)
        C_two5,Y_two5 = N_DUDE.make_data_for_Two_NN_DUDE(P[4],Z[4*n:(4+1)*n],k,
                                                       L_new[4*alpha_size:(4+1)*alpha_size,],nb_classes,n,offset)
        C_two6,Y_two6 = N_DUDE.make_data_for_Two_NN_DUDE(P[5],Z[5*n:(5+1)*n],k,
                                                       L_new[5*alpha_size:(5+1)*alpha_size,],nb_classes,n,offset)
        C_two7,Y_two7 = N_DUDE.make_data_for_Two_NN_DUDE(P[6],Z[6*n:(6+1)*n],k,
                                                       L_new[6*alpha_size:(6+1)*alpha_size,],nb_classes,n,offset)
        C_two8,Y_two8 = N_DUDE.make_data_for_Two_NN_DUDE(P[7],Z[7*n:(7+1)*n],k,
                                                       L_new[7*alpha_size:(7+1)*alpha_size,],nb_classes,n,offset)
        C_two9,Y_two9 = N_DUDE.make_data_for_Two_NN_DUDE(P[8],Z[8*n:(8+1)*n],k,
                                                       L_new[8*alpha_size:(8+1)*alpha_size,],nb_classes,n,offset)
        C_two10,Y_two10 = N_DUDE.make_data_for_Two_NN_DUDE(P[9],Z[9*n:(9+1)*n],k,
                                                       L_new[9*alpha_size:(9+1)*alpha_size,],nb_classes,n,offset)
        
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
        model.fit(C_two10,Y_two10,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two10, Y_two10))
        model.fit(C_two9,Y_two9,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two9, Y_two9))
        model.fit(C_two8,Y_two8,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two8, Y_two8))
        model.fit(C_two7,Y_two7,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two7, Y_two7))
        model.fit(C_two6,Y_two6,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two6, Y_two6))
        model.fit(C_two5,Y_two5,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two5, Y_two5))
        model.fit(C_two4,Y_two4,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two4, Y_two4))
        model.fit(C_two3,Y_two3,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two3, Y_two3))
        model.fit(C_two2,Y_two2,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two2, Y_two2))
        model.fit(C_two1,Y_two1,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two1, Y_two1))
        model.fit(C_two10,Y_two10,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two10, Y_two10))
        model.fit(C_two9,Y_two9,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two9, Y_two9))
        model.fit(C_two8,Y_two8,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two8, Y_two8))
        model.fit(C_two7,Y_two7,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two7, Y_two7))
        model.fit(C_two6,Y_two6,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two6, Y_two6))
        model.fit(C_two5,Y_two5,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two5, Y_two5))
        model.fit(C_two4,Y_two4,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two4, Y_two4))
        model.fit(C_two3,Y_two3,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two3, Y_two3))
        model.fit(C_two2,Y_two2,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two2, Y_two2))
        model.fit(C_two1,Y_two1,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two1, Y_two1))
        model.fit(C_two10,Y_two10,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two10, Y_two10))
        model.fit(C_two9,Y_two9,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two9, Y_two9))
        model.fit(C_two8,Y_two8,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two8, Y_two8))
        model.fit(C_two7,Y_two7,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two7, Y_two7))
        model.fit(C_two6,Y_two6,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two6, Y_two6))
        model.fit(C_two5,Y_two5,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two5, Y_two5))
        model.fit(C_two4,Y_two4,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two4, Y_two4))
        model.fit(C_two3,Y_two3,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two3, Y_two3))
        model.fit(C_two2,Y_two2,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two2, Y_two2))
        model.fit(C_two1,Y_two1,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two1, Y_two1))
        model.fit(C_two10,Y_two10,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two10, Y_two10))
        model.fit(C_two9,Y_two9,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two9, Y_two9))
        model.fit(C_two8,Y_two8,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two8, Y_two8))
        model.fit(C_two7,Y_two7,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two7, Y_two7))
        model.fit(C_two6,Y_two6,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two6, Y_two6))
        model.fit(C_two5,Y_two5,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two5, Y_two5))
        model.fit(C_two4,Y_two4,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two4, Y_two4))
        model.fit(C_two3,Y_two3,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two3, Y_two3))
        model.fit(C_two2,Y_two2,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two2, Y_two2))
        model.fit(C_two1,Y_two1,nb_epoch=10,batch_size=100,show_accuracy=True, verbose=0, validation_data=(C_two1, Y_two1))
        
        pred_class_two=model.predict_classes(C_two1, batch_size=200, verbose=0)
        s_nn_hat_two=N_DUDE.mapping_mat_resize(pred_class_two,k,n)
        x_nn_hat_two=N_DUDE.denoise_with_s_Two_NN_DUDE(z[i],s_nn_hat_two) 
        error_nn_two=N_DUDE.error_rate(x,x_nn_hat_two)
        print '2-D N-DUDE =', error_nn_two
        Error_Two_NN_DUDE[0,k]=error_nn_two
        X_hat_Two_NN_DUDE[k_max*0+k-1,:]=x_nn_hat_two
        
        One_NN_End=time.time()
        One_NN_Duration=One_NN_End-One_NN_Start
        
        print 'Time =', One_NN_Duration
        print "---------------------------------------------------"
        
        res_file='/HDD/user/yoon/Yoon_SV4/N-DUDE_SV4/NeuralDUDE_Delta_Variation/Result_Plot/Fine_Tune_Two_ver5_6'
        np.savez(res_file, Error_Two_DUDE=Error_Two_DUDE,
                 Error_Two_NN_DUDE=Error_Two_NN_DUDE,
                 X_hat_Two_NN_DUDE=X_hat_Two_NN_DUDE)