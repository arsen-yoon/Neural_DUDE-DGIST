import numpy as np
import os
import re
import time
    
from numpy import *
from operator import itemgetter, attrgetter, methodcaller

## regular expression for header parsing ##
re1 = re.compile('ERCC-[0-9]{5}')
re2 = re.compile('[0-9]+')
re3 = re.compile('\*|([0-9]+[MIDNSHPX=])+')
re4 = re.compile('[A-T]{100}')
num_head = 6

def sam_info(root,num_sam,num_fa,num_line):
    for i in range(0,1):
        path = os.path.join(root, 'ERCC%d.fa' %num_fa)
    
    clean_data = open(path,'r')
    clean_seq = clean_data.readlines()
    clean_data.close()
    
    ## make clean dict {'Spike':clean_seq} ##
    spike_idx = [] # to take idx of spike name line #
    clean_dict = {}
    
    for i in range(len(clean_seq)):
        c = re1.search(clean_seq[i])
        if c != None:
            spike_idx += [i]
            if not clean_dict.has_key(c.group()):
                clean_dict.update({c.group():''})
    
    spike_idx += [len(clean_seq)]

    for j in range(len(spike_idx)-1):
        con_clean = ''
        for k in range(spike_idx[j]+1,spike_idx[j+1]):
            con_clean += clean_seq[k][:-1]
        clean_dict[re1.search(clean_seq[spike_idx[j]]).group()]=con_clean
    
    
    for i in range(0,1): # for many files, num_sam can be array(or list)
        path = os.path.join(root, 'ERR%d.sam' %num_sam)
    if num_line != 0:
        noisy_data = open(path,'r')
        noisy_seq = noisy_data.readlines()[94:num_line+94]
        noisy_data.close()
    elif num_line == 0:
        noisy_data = open(path,'r')
        noisy_seq = noisy_data.readlines()[94:]
        noisy_data.close()
        
    ## make header array ##
    
    err_count=0
    HD1 = []
    HD2 = []
    HD3 = []
    HD4 = []
    HD5 = []
    HD6 = []
    
    print "Pre-processing starts..."
    for j in range(len(noisy_seq)):
        if j%100000==0:
            print "Pre-processing is %0.1f%% done.."%(j/float(len(noisy_seq))*100.)
        tmp = noisy_seq[j]
        head1 = re1.search(tmp) # ref. spike name
        head2 = re2.findall(tmp) # match pos. of clean_seq
        head3 = re3.search(tmp) # 30S70M
        head4 = re4.search(tmp) # noisy_seq. length=100
        
        cigar_str = head3.group()
        noisy = head4.group()
        
        if 'D' in cigar_str or 'I' in cigar_str or '*' in cigar_str: # Deletion & Insertion error pop up
            err_count += 1
        elif head1==None: # Error seq pop up
            err_count += 1
        elif 'N' in noisy: # Error seq pop up
            err_count += 1
        else:
            r_name = head1.group()
            tot_seq = clean_dict[r_name]
            tmp_seq = tot_seq[int(head2[4])-1:int(head2[4])-1+100] # for 1-based.
            m_idx = cigar_str.index('M')
            # clear till this
            
            if m_idx == 1: # *M
                match_idx = int(cigar_str[m_idx-1])
                noisy = noisy[:match_idx]
                cut_seq = tmp_seq[:match_idx]
            elif m_idx == 2: # **M
                match_idx = int(cigar_str[m_idx-2]+cigar_str[m_idx-1])
                noisy = noisy[:match_idx]
                cut_seq = tmp_seq[:match_idx]
            elif m_idx == 3: 
                if cigar_str.count('S'): # *S*M
                    s_idx = cigar_str.index('S')
                    match_idx = int(cigar_str[s_idx+1])
                    ss_idx = int(cigar_str[s_idx-1])
                    noisy = noisy[ss_idx:ss_idx+match_idx]
                    cut_seq = tmp_seq[:match_idx]
                else:
                    match_idx = 100 # ***M
                    noisy = noisy[:match_idx]
                    cut_seq = tmp_seq[:match_idx]
            elif m_idx == 4: 
                if cigar_str.count('S'): 
                    s_idx = cigar_str.index('S')
                    if s_idx == 1: # *S**M*S
                        match_idx = int(cigar_str[2]+cigar_str[3])
                        ss_idx = int(cigar_str[0])
                        noisy = noisy[ss_idx:ss_idx+match_idx]
                        cut_seq = tmp_seq[:match_idx]
                    elif s_idx == 2: # **S*M
                        match_idx = int(cigar_str[s_idx+1])
                        ss_idx = int(cigar_str[s_idx-2]+cigar_str[s_idx-1])
                        noisy = noisy[ss_idx:ss_idx+match_idx]
                        cut_seq = tmp_seq[:match_idx]
            elif m_idx == 5: # **S**M
                if cigar_str.count('S'):
                    s_idx = cigar_str.index('S')
                    match_idx = int(cigar_str[s_idx+1]+cigar_str[s_idx+2])
                    ss_idx = int(cigar_str[s_idx-2]+cigar_str[s_idx-1])
                    noisy = noisy[ss_idx:ss_idx+match_idx]
                    cut_seq = tmp_seq[:match_idx]
            else:
                print "There is another case!!"
                break
               
            HD1.append(r_name)
            HD2.append(head2[4])
            HD3.append(cigar_str)
            HD4.append(cut_seq)
            HD5.append(noisy)
            HD6.append(len(cut_seq))
            
    print "Reshaping..."
    
    reshaped1 = np.reshape(HD1, (len(noisy_seq)-err_count, 1))
    reshaped2 = np.reshape(HD2, (len(noisy_seq)-err_count, 1))
    reshaped3 = np.reshape(HD3, (len(noisy_seq)-err_count, 1))
    reshaped4 = np.reshape(HD4, (len(noisy_seq)-err_count, 1))
    reshaped5 = np.reshape(HD5, (len(noisy_seq)-err_count, 1))
    reshaped6 = np.reshape(HD6, (len(noisy_seq)-err_count, 1))
    
    new_arr = np.hstack((reshaped1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped6))
    new_arr = sorted(new_arr, key=itemgetter(0))
    new_arr = np.reshape(new_arr, (len(noisy_seq)-err_count, num_head))
    print "Pre-processing is completed!!"
    return new_arr # list of strs

def make_PI(new_arr):
    PI = zeros((4,4))
    dict_A = {'A':0, 'T':0, 'G':0, 'C':0}
    dict_T = {'A':0, 'T':0, 'G':0, 'C':0}
    dict_G = {'A':0, 'T':0, 'G':0, 'C':0}
    dict_C = {'A':0, 'T':0, 'G':0, 'C':0}
    tot_dict = [dict_A, dict_C, dict_G, dict_T]
    print "Making PI starts..."
    for i in range(len(new_arr)):
        if i%100000==0:
            print "Making PI is %0.1f%% done.."%(i/float(len(new_arr))*100.) 
        for j in range(len(new_arr[i][3])):
            if new_arr[i][3][j] == 'A':
                if new_arr[i][4][j] == 'A':
                    dict_A[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'T':
                    dict_A[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'G':
                    dict_A[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'C':
                    dict_A[new_arr[i][4][j]] += 1
                else:
                    print "Error 1"
                    break
            
            elif new_arr[i][3][j] == 'C':
                if new_arr[i][4][j] == 'A':
                    dict_C[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'T':
                    dict_C[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'G':
                    dict_C[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'C':
                    dict_C[new_arr[i][4][j]] += 1
                else:
                    print "Error 2"
                    break
            
            elif new_arr[i][3][j] == 'G':
                if new_arr[i][4][j] == 'A':
                    dict_G[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'T':
                    dict_G[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'G':
                    dict_G[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'C':
                    dict_G[new_arr[i][4][j]] += 1
                else:
                    print "Error 3"
                    break
            
            elif new_arr[i][3][j] == 'T':
                if new_arr[i][4][j] == 'A':
                    dict_T[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'T':
                    dict_T[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'G':
                    dict_T[new_arr[i][4][j]] += 1
                elif new_arr[i][4][j] == 'C':
                    dict_T[new_arr[i][4][j]] += 1
                else:
                    print "Error 4"
                    break
            else:
                print "Error 5"
                break
    
    for p in range(PI.shape[0]):
        PI[p,0] = tot_dict[p]['A']
        PI[p,1] = tot_dict[p]['C']
        PI[p,2] = tot_dict[p]['G']
        PI[p,3] = tot_dict[p]['T']
    for q in range(PI.shape[0]):
        PI[q] = PI[q]/float(sum(PI[q]))
    print "Making PI is completed!!"
    return PI