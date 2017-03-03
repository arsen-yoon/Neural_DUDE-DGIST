import Parsing as ps
import numpy as np
import re
import os
os.environ["THEANO_FLAGS"]="mode=FAST_RUN, device=cpu,floatX=float32"
import tetra_dude as td
import matplotlib.pyplot as plt
import datetime

from operator import itemgetter, attrgetter, methodcaller
from numpy import *

import keras

from keras.preprocessing import sequence
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.utils import np_utils

new = ps.sam_info('./',1211177, 92, 100) #For all lines, num_line(last) = 0

#ATGC
PI = array([[0.996975631, 0.000512946175, 0.00151638794, 0.000995034679],
            [0.000416377636, 0.997385479, 0.000858583848, 0.00133955915],
            [0.000865811030, 0.000725518163, 0.997926104, 0.000482566345],
            [0.000634619373, 0.000847625845, 0.000434391514, 0.998083363]])
'''
#ACGT
PI = array([[0.996975631, 0.000995034679, 0.00151638794, 0.000512946175],
            [0.000634619373, 0.998083363, 0.000434391514, 0.000847625845],
            [0.000865811030, 0.000482566345, 0.997926104, 0.000725518163],
            [0.000416377636, 0.00133955915, 0.000858583848, 0.997385479]])
'''
with open("fold1/Illumina_LinErr_100_fold1_test1.fasta", "w") as f:
    for i in range(len(new)):
        data = "%s\n"%new[i][4] # determine whether the data is made as line by line or connected one line using \n
        f.write(data)

with open("fold1/Noisy_test.fasta", "w") as f:
    for i in range(len(new)):
        data = "%s"%new[i][4] 
        f.write(data)
        
with open("fold1/Clean_test1.fasta","w") as f:
    for i in range(len(new)):
        data = "%s\n"%new[i][3]
        f.write(data)
        
with open("fold1/Clean_test2.fasta","w") as f:
    for i in range(len(new)):
        data = "%s"%new[i][3]
        f.write(data)
        
nb_classes = 4
nt_order = "ATGC"
H = linalg.inv(PI)
LAMBDA = array([[0, 1, 1, 1],
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [1, 1, 1, 0]])
S = zeros((nb_classes, nb_classes**nb_classes),dtype=int)
for s in range(1,nb_classes**nb_classes):
    for x in range(4):
        S[x][s] = S[x][s-1]
    
    if S[0][s] != 3:
        S[0][s] += 1
        continue
    else:
        S[0][s] = 0
        
    if S[1][s] != 3:
        S[1][s] += 1
        continue
    else:
        S[1][s] = 0
        
    if S[2][s] != 3:
        S[2][s] += 1
        continue
    else:
        S[2][s] = 0
        
    if S[3][s] != 3:
        S[3][s] += 1
        continue
    else:
        print "ERROR: S index out of bound!\n"
RHO   = zeros((nb_classes, nb_classes**nb_classes),dtype=float)

RHO_R = zeros((nb_classes, nb_classes*nb_classes),dtype=float)

for s in range(nb_classes**nb_classes):
    for x in range(4):
        RHO[x][s] = (PI[x][0]*LAMBDA[x][S[0][s]] + PI[x][1]*LAMBDA[x][S[1][s]] +
                     PI[x][2]*LAMBDA[x][S[2][s]] + PI[x][3]*LAMBDA[x][S[3][s]])
for s in range(nb_classes*nb_classes):
    for x in range(4):
        RHO_R[x][s] = PI[x][int(s/4)]*LAMBDA[x][s%4]
L=dot(H,RHO) 
L_new=-L+amax(L)     # A new loss matrix
L_R = dot(H,RHO_R)
L_R_new=-L_R+amax(L_R)
k_max=26


### Array for save ###
Error_One_NN = zeros((k_max+1,))
Error_One_NN[0] = 1
Error_One_NN_R = zeros((k_max+1,))        
Error_One_NN_R[0] = 1

LOG = open("log.txt","w")
fold_name = ['fold1']#, 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9']
file_stub = 'Illumina_LinErr_'
file_name= []
for i in range(len(fold_name)):
    for j in range(2,10):
            continue
            name = '%s/%s0%s0_%s_test1' % (fold_name[i], file_stub, str(j), fold_name[i])
            file_name.append(name)
    name = '%s/%s100_%s_test1' % (fold_name[i], file_stub, fold_name[i])
    file_name.append(name)

for k in range(1,15):
    print "k=%d"%k
    for name in file_name:
        name_out = name + '_ND' + str(k)
        f_in = open("%s.fasta" % name, "r")
        lines = f_in.readlines()
        f_in.close()
        
        ### 1-D DUDE ###
        f_out1 = open("%s_1.fasta" % name_out, "w")
        seq,C,Y,Y_R = td.make_data_for_ndude_new2(lines,k,L_new,L_R_new,nt_order)
        td.dude(f_out1,seq,H,LAMBDA,PI,k,nt_order)
        f_out1.close()
        f_out11 = open("%s_1.fasta" % name_out, "r")
        result1 = f_out11.readlines()
        f_out11.close()
        f_out22 = open("fold1/Clean_test2.fasta","r")
        cl_seq = f_out22.readlines()
        f_out22.close()
        error_dude=td.error_rate_new3(cl_seq[0],result1[0])
        print '1-D DUDE=',error_dude
        
        ### 1-D N-DUDE ###
        
        
        seq,C,Y,Y_R = td.make_data_for_ndude_new3(lines,k,L_new,L_R_new,nt_order)
        
        model1=Sequential()
        model1.add(Dense(40,input_dim=2*k*(nb_classes),init='he_normal'))
        model1.add(Activation('relu'))
        model1.add(Dense(40,init='he_normal'))
        model1.add(Activation('relu'))
        model1.add(Dense(40,init='he_normal'))
        model1.add(Activation('relu'))
        model1.add(Dense(256,init='he_normal'))
        model1.add(Activation('softmax'))

        rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06,clipnorm=1.5)
        adagrad=Adagrad(clipnorm=1.5)
        adam=Adam()
        adadelta=Adadelta()
        sgd=SGD(lr=0.01,decay=1e-6,momentum=0.95, nesterov=True, clipnorm=1.0)

        model1.compile(loss='poisson', optimizer=adam)
        model1.fit(C,Y,nb_epoch=10,batch_size=128, show_accuracy=False, verbose=1)
        
        f_out1 = open("%s_1.fasta" % name_out, "w")
        pred_class1=model1.predict_classes(C, batch_size=128, verbose=0)
        print "pred_class1:", pred_class1, pred_class1.shape
        td.denoise_with_s_new3(f_out1,seq,pred_class1,k,nt_order)
        f_out1.close()
        f_out11 = open("%s_1.fasta" % name_out, "r")
        result1 = f_out11.readlines()
        f_out11.close()
        
        f_out11 = open("fold1/Noisy_test.fasta", "r")
        to_test = f_out11.readlines()
        f_out11.close()
        
        f_out22 = open("fold1/Clean_test2.fasta","r")
        cl_seq = f_out22.readlines()
        f_out22.close()
        #print "Noisy:", to_test[0], len(to_test[0])
        #print "Denoised:", result1[0], len(result1[0])
        #print "Clean:", cl_seq[0], len(cl_seq[0])
        error_nn1 = td.error_rate_new3(cl_seq[0],result1[0])
        error_nn2 = td.error_rate_new3(cl_seq[0],to_test[0])
        Error_One_NN[k] = error_nn1
        print "Denoised/Clean:",error_nn1
        print "Noisy/Clean:",error_nn2
        print "Denoised/Noisy:",error_nn1/float(error_nn2)
       
        model2=Graph()
        model2.add_input(name='input', input_shape=(2*k*(nb_classes),))
        model2.add_node(Dense(400,init='he_normal'), name='layer1', input='input')
        model2.add_node(Activation('relu'), name='activation1', input='layer1')
        model2.add_node(Dense(400,init='he_normal'), name='layer2', input='activation1')
        model2.add_node(Activation('relu'), name='activation2', input='layer2')
        model2.add_node(Dense(400,init='he_normal'), name='layer3', input='activation2')
        model2.add_node(Activation('relu'), name='activation3', input='layer3')
        model2.add_node(Dense(4,init='he_normal'), name='layer4_0', input='activation3')
        model2.add_node(Dense(4,init='he_normal'), name='layer4_1', input='activation3')
        model2.add_node(Dense(4,init='he_normal'), name='layer4_2', input='activation3')
        model2.add_node(Dense(4,init='he_normal'), name='layer4_3', input='activation3')
        model2.add_node(Activation('softmax'), name='activation4_0', input='layer4_0')
        model2.add_node(Activation('softmax'), name='activation4_1', input='layer4_1')
        model2.add_node(Activation('softmax'), name='activation4_2', input='layer4_2')
        model2.add_node(Activation('softmax'), name='activation4_3', input='layer4_3')
        model2.add_output(name='output0', input='activation4_0')
        model2.add_output(name='output1', input='activation4_1')
        model2.add_output(name='output2', input='activation4_2')
        model2.add_output(name='output3', input='activation4_3')

        model2.compile(loss={'output0':'poisson','output1':'poisson','output2':'poisson','output3':'poisson'}, 
                       optimizer=adam)

        model2.fit({'input':C,'output0':Y_R[:,0:4],'output1':Y_R[:,4:8],'output2':Y_R[:,8:12],'output3':Y_R[:,12:16]}, 
                   nb_epoch=1,batch_size=,verbose=1)
        
    # -----------------------------------------------------
        f_out2 = open("%s_2.fasta" % name_out, "w")
        pred2=model2.predict({'input':C}, batch_size=16384, verbose=0)
        pred_class2 = td.s_R_preprocess(pred2, k)
        td.denoise_with_s_R(f_out2,lines,pred_class2,k,nt_order)
        f_out2.close()
        f_out22 = open("%s_2.fasta" % name_out, "r")
        result2 = f_out22.readlines()
        f_out22.close()
        f_out33 = open("fold1/Clean_test2.fasta","r")
        cl_seq = f_out33.readlines()
        f_out33.close()
        error_nn_r = td.error_rate_new3(result2[0],cl_seq[0])
        
        Error_One_NN_R[k] = error_nn_r
        print error_nn_r
        
        res_file='fold1/Test1'
        np.savez(res_file,Error_One_NN=Error_One_NN,Error_One_NN_R=Error_One_NN_R)