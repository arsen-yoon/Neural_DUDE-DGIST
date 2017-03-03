
# coding: utf-8

# In[1]:
import numpy as np
from numpy import *
import datetime
    
# In[4]:
          
def compare(a,b):
    if a==b:
        e = 0
    else:
        e = 1
    return e

def error_rate_new2(a,b):
    error=zeros(len(a)-1)
    for i in range(len(a)-1):
        error[i]=compare(a[i],b[i])
    return sum(error)

def error_rate_new3(a,b):
    error=zeros(len(a))
    for i in range(len(a)):
        error[i]=compare(a[i],b[i])
    return sum(error)/len(a)

def make_data_for_ndude(lines,k,L,L_R,nt_order): # Original
    C = zeros((150*int(len(lines)/2),2*k*4),dtype=int) # Context mat. num of classes = 4 (alphabet size)
    Z = zeros((150*int(len(lines)/2),4),dtype=int)
    czn = 0
    for t in range(int(len(lines)/2)):
        print t
        Z_in = zeros((len(lines[t*2+1]),6),dtype=int)
        for i in range(len(lines[t*2+1])):
            for j in range(6):
                if lines[t*2+1][i] == nt_order[j]:
                    Z_in[i,j] = 1
                    lines[t*2+1] = lines[t*2+1][:i] + str(j) + lines[t*2+1][i+1:]
                    break
    
        for i in range(k,len(lines[t*2+1])-k-1):
            if sum(Z_in[i-k:i+k+1,4]) > 0:
                continue
    
            c_i = vstack((Z_in[i-k:i,0:4],Z_in[i+1:i+k+1,0:4])).reshape(1,2*k*4)
            C[czn,] = c_i
            Z[czn,] = Z_in[i,0:4]
            czn += 1
    print "tmp start"
    Y=dot(Z[:czn,],L)                    #Original
    print "tmp2 start"
    Y_R=dot(Z[:czn,],L_R)                #Reduced
    
    return lines, C[:czn,], Y, Y_R

def make_data_for_ndude_new(lines,k,L,L_R,nt_order,tot_syms): # Use arrays instead of lists
    C = zeros((tot_syms-len(lines)*2*k,2*k*4),dtype=int)
    Z = zeros((tot_syms-len(lines)*2*k,4),dtype=int)
    count = 0
    for i in range(len(lines)):
        print i
        z_i = zeros((len(lines[i]),4),dtype=int)
        for j in range(len(lines[i])):
            for r in range(4):
                if lines[i][j] == nt_order[r]:
                    lines[i] = lines[i][:j] + str(r) + lines[i][j+1:] # mapping the alphabet to num. A = 0, T = 1, G = 2, C = 3
                    z_i[j,r] = 1
                    break
        for p in range(k,len(lines[i])-k):
            c_i = vstack((z_i[p-k:p,],z_i[p+1:p+k+1,])).reshape(1,2*k*4)
            C[count,] = c_i
            Z[count,] = z_i[p,]
            count += 1
    
    print "tmp start"
    Y=dot(Z,L)                    #Original
    print "tmp2 start"
    Y_R=dot(Z,L_R)                #Reduced
    
    return lines, C, Y, Y_R

def make_data_for_ndude_new2(lines,k,L,L_R,nt_order): # Line by line ver.
    C = []
    Z = []
    for i in range(len(lines)):
    #    print "Alphabet seq:",lines[i], len(lines[i])
        lines[i] = lines[i][:-1]
        z_i = zeros((len(lines[i]),4),dtype=int) # for not consider the last \n, -1
        for j in range(len(lines[i])):
            for r in range(4):
                if lines[i][j] == nt_order[r]:
                    lines[i] = lines[i][:j] + str(r) + lines[i][j+1:]
                    z_i[j,r] =1
                    break
    #        print "Mapping seq:",lines[i], len(lines[i])

        for p in range(k,len(lines[i])-k): # \n is excluded above. 
            c_i = vstack((z_i[p-k:p,],z_i[p+1:p+k+1,])).reshape(1,2*k*4)
    #        print "c_i:", c_i
            C.append(c_i)
            Z.append(z_i[p,])
    #        print "z_i[p,]", z_i[p,]
    
    C = np.reshape(C,(-1,2*k*4))
    #print "C:",C
    Z = np.reshape(Z,(-1,4))
    #print "Z:",Z
    Y = zeros((Z.shape[0],L.shape[1]),dtype=int)
    Y_R = zeros((Z.shape[0],L_R.shape[1]),dtype=int)
    for i in range(Y.shape[0]):
        Y[i] = dot(Z[i],L)
        Y_R[i] = dot(Z[i],L_R)
    #print "Y:",Y
    #print "Y_R:",Y_R
    #for i in range(len(lines)):
    #    print "lines:", lines[i], len(lines[i])
    return lines, C, Y, Y_R

def make_data_for_ndude_new3(lines,k,L,L_R,nt_order): # Connected ver.
    seq = lines[0]
    tot_syms = len(seq)
    C = zeros((tot_syms-2*k,2*k*4),dtype=int)
    Z = zeros((tot_syms,4),dtype=int)
    Y = zeros((tot_syms-2*k,4**4),dtype=int)
    Y_R = zeros((tot_syms-2*k,4*4),dtype=int)
    for i in range(tot_syms):
        for j in range(4):
            if seq[i] == nt_order[j]:
                seq = seq[:i] + str(j) + seq[i+1:]
    #        print "Mapping seq:", seq, len(seq)
            
        Z[i,int(seq[i])] = 1
    #print "Z:",Z, Z.shape
    for i in range(k,tot_syms-k):
        c_i = vstack((Z[i-k:i],Z[i+1:i+k+1])).reshape(1,2*k*4)
        #print "C_i:", c_i, c_i.shape
        C[i-k,] = c_i
        Y[i-k,] = dot(Z[i],L)
        Y_R[i-k,] = dot(Z[i],L_R)
    #print "seq:",seq, len(seq)
    #print "C:",C, C.shape
    #print "Y:",Y, Y.shape
    #print "Y_R",Y_R, Y_R.shape
    #print C, C.shape,"\n"
    #print Y, Y.shape,"\n"
    #print Y_R, Y_R.shape
    #print seq, len(seq)
    return seq, C, Y, Y_R

def make_data_for_ndude_new4(lines,k,L,L_R,nt_order): # Connected ver.
    seq = lines[0]
    tot_syms = len(seq)
    C = zeros((tot_syms-2*k,2*k*4),dtype=int)
    Z = zeros((tot_syms,4),dtype=int)
    Y = zeros((tot_syms-2*k,4**4),dtype=int)
    Y_R = zeros((tot_syms-2*k,4*4),dtype=int)
    Z_tmp = zeros((2*k+1,4),dtype=int)
    for i in range(k,tot_syms-k):
        print i
        for l in range(i-k,i+k+1):
            for j in range(4):
                if seq[l] == nt_order[j]:
                    seq = seq[:l] + str(j) + seq[l+1:]
            Z_tmp[l-(i-k),int(seq[l])] = 1
         
    for i in range(k,tot_syms-k):
        print i
        c_i = vstack((Z[i-k:i],Z[i+1:i+k+1])).reshape(1,2*k*4)
        C[i-k,] = c_i
        Y[i-k,] = dot(Z[i],L)
        Y_R[i-k,] = dot(Z[i],L_R)
       
    #print C, C.shape,"\n"
    #print Y, Y.shape,"\n"
    #print Y_R, Y_R.shape
    #print seq, len(seq)
    return seq, C, Y, Y_R

'''
def dude(f,lines,H,LAMBDA,PI,k,nt_order):
    lines_new = list(lines)
    m={}
    
    for t in range(int(len(lines)/2)):
        for i in range(k,len(lines[t*2+1])-k-1):
            if lines[t*2+1][i-k:i+k+1].count('4') > 0 :
                continue
            context_str =lines[t*2+1][i-k:i] + lines[t*2+1][i+1:i+k+1]
            if not m.has_key(context_str):
                m[context_str]=zeros(4,dtype=int)
                m[context_str][int(lines[t*2+1][i])]=1
            else:
                m[context_str][int(lines[t*2+1][i])]+=1
    
    for t in range(len(lines)):
        if t % 2 == 0:
            f.write(lines_new[t])
            continue
        for i in range(len(lines[t])):
            if i < k or len(lines[t])-k-1 <= i or lines[t][i-k:i+k+1].count('4') > 0 :
                f.write(nt_order[int(lines[t][i])])
            else:
                context_str =lines[t][i-k:i] + lines[t][i+1:i+k+1]
                S_err=zeros(4,dtype=float)
                for j in range(4):
                    S_err[j] = dot(dot(m[context_str],H),LAMBDA[:,j]*PI[:,int(lines[t][i])])  
                f.write(nt_order[argmin(S_err)])
'''
def dude(f,lines,H,LAMBDA,PI,k,nt_order):
    lines_new = list(lines)
    m={}
    for t in range(int(len(lines))):
        for i in range(k,len(lines[t])-k):
            context_str =lines[t][i-k:i] + lines[t][i+1:i+k+1]
            if not m.has_key(context_str):
                m[context_str]=zeros(4,dtype=int)
                m[context_str][int(lines[t][i])]=1
            else:
                m[context_str][int(lines[t][i])]+=1
    
    for t in range(len(lines)):
        for i in range(len(lines[t])):
            if i < k or len(lines[t])-k <= i:
                f.write(nt_order[int(lines[t][i])])
            else:
                context_str =lines[t][i-k:i] + lines[t][i+1:i+k+1]
                S_err=zeros(4,dtype=float)
                for j in range(4):
                    S_err[j] = dot(dot(m[context_str],H),LAMBDA[:,j]*PI[:,int(lines[t][i])])  
                f.write(nt_order[argmin(S_err)])

def denoise_with_s(f,lines,s,k,nt_order): # For original
    lines_new = list(lines)
    sn = 0
    for t in range(len(lines)):
        if t % 2 == 0:
            f.write(lines_new[t])
            continue
                      
        for i in range(len(lines[t])):
            if i < k or len(lines[t])-k-1 <= i or lines[t][i-k:i+k+1].count('4') > 0 :
                f.write(nt_order[int(lines[t][i])])
            else:
                s_index = s[sn]
                sn += 1
                DN = zeros(4,dtype=int)
                DN[3] = int(s_index / (4**3))
                s_index -= DN[3] * (4**3)
                DN[2] = int(s_index / (4**2))
                s_index -= DN[2] * (4**2)
                DN[1] = int(s_index / (4**1))
                s_index -= DN[1] * (4**1)
                DN[0] = s_index / (4**0)
                f.write(nt_order[DN[int(lines[t][i])]])

def denoise_with_s_new3(f,lines,s,k,nt_order): # For connected ver.
    sn = 0
    for i in range(len(lines)):
        if i < k or len(lines)-k <= i:
                f.write(nt_order[int(lines[i])])
        else:
            s_index = s[sn]
            sn += 1
            DN = zeros(4,dtype=int)                # ex) s_index = 212, noisy = 'G' = 2
            DN[3] = int(s_index / (4**3))          #     DN[3] = int(212/64) = 3
            s_index -= DN[3] * (4**3)              #     s_index = 212 - 3*64 = 20 (remainder)
            DN[2] = int(s_index / (4**2))          #     DN[2] = int(20/16) = 1
            s_index -= DN[2] * (4**2)              #     s_index = 20 - 16 = 4
            DN[1] = int(s_index / (4**1))          #     DN[1] = int(4/4) = 1
            s_index -= DN[1] * (4**1)              #     s_index = 0
            DN[0] = s_index / (4**0)               #     DN[0] = int(0/1) = 0 -> DN = [0 1 1 3]
            f.write(nt_order[DN[int(lines[i])]])   #     nt_order[DN[int(lines[i])]] = nt_order[DN[2]] = nt_order[1] = T

def denoise_with_s_new2(f,lines,s,k,nt_order): # For line by line ver.
    sn = 0
    for t in range(len(lines)):
        for i in range(len(lines[t])):
            if i < k or len(lines[t])-k <= i:
                f.write(nt_order[int(lines[t][i])])
            else:
                s_index = s[sn]
                sn += 1
                DN = zeros(4,dtype=int)
                DN[3] = int(s_index / (4**3))
                s_index -= DN[3] * (4**3)
                DN[2] = int(s_index / (4**2))
                s_index -= DN[2] * (4**2)
                DN[1] = int(s_index / (4**1))
                s_index -= DN[1] * (4**1)
                DN[0] = s_index / (4**0)
                f.write(nt_order[DN[int(lines[t][i])]])

def denoise_with_s_R(f,lines,s,k,nt_order):
    sn = 0
    for t in range(len(lines)):
        for i in range(len(lines[t])):
            S = zeros(4)
            for j in range(4):
                S[j] = s[sn][int(lines[t][i])*4+j]      
            f.write(nt_order[argmax(S)])
            sn+=1

def s_R_preprocess(s_pre, k):
    n = s_pre['output0'].shape[0]
    s_nn_R_hat=zeros((n, 16))
    for i in range(n):
        s_i_temp = zeros(16,dtype=int)
                      
        if max(s_pre['output0'][i]) == s_pre['output0'][i][0]:
            s_i_temp[0] = 1
        else:
            s_i_temp[argmax(s_pre['output0'][i])] = 1

        if max(s_pre['output1'][i]) == s_pre['output1'][i][1]:
            s_i_temp[5] = 1
        else:
            s_i_temp[argmax(s_pre['output1'][i])+4] = 1

        if max(s_pre['output2'][i]) == s_pre['output2'][i][2]:
            s_i_temp[10] = 1
        else:
            s_i_temp[argmax(s_pre['output2'][i])+8] = 1

        if max(s_pre['output3'][i]) == s_pre['output3'][i][3]:
            s_i_temp[15] = 1
        else:
            s_i_temp[argmax(s_pre['output3'][i])+12] = 1

        s_i = hstack(s_i_temp)
        s_nn_R_hat[i,]=s_i
    return s_nn_R_hat


def PRINT(f,s):
    out = str(datetime.datetime.now()) + '\t' + s
    print out
    f.write(out+'\n')


