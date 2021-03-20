# State - Space AFCM initiative
# No time iterations
import math
import numpy as np
from copy import deepcopy
import pandas as pd


import os


def disturbance_row (row_before,weight_matrix): # a list describing the new values of the concepts if we perfomred classif FCM
    dc_row = []
    
    for i in range (0,len(row_before)):
        new_value_of_i = 0

        disturbance_total = 0
        for k in range (0,len(row_before)-1):
            disturbance_total = disturbance_total + row_before[k]*weight_matrix[k,i]
        new_i = row_before[i] + disturbance_total
        dc_row.append(new_i)
    
    dc_row = np.array(dc_row)
    #dc_row = dc_row - np.array(row_before)
    #print ('DC_ROW : {}'.format(dc_row))
    return dc_row
            



def calc_new_value_of_concept (attribute_number, weight_matrix, dist_matr, inputs,activation): # t is an integer describing which concept we are calculating, w is the weight matrix, instance is the row with the initial values of the concepts


    concept_disturbance = dist_matr[attribute_number-1]
    #if attribute_number == 32:
        #print('Disturbance: {}'.format(concept_disturbance))
    
    sumw = 0 
    for i in range (len(inputs)-1):
        
        sumw = sumw + abs( weight_matrix[i, attribute_number-1])
    #if attribute_number == 32:
        #print(sumw)
    #print ('SumW:{}'.format(sumw))
    
    if concept_disturbance==0:
        new_value = inputs[attribute_number-1] # if no disturbance, return null
        #print ('New value: {}'.format(new_value))
    else:
        if sumw>1 or sumw<-1:
            new_value = inputs[attribute_number-1] + (concept_disturbance/abs(sumw))
        else:
            new_value = inputs[attribute_number-1] + concept_disturbance

    #if attribute_number == 32:
        #print(new_value)
            
        if activation == 'sigmoid':
            #new_value = -1 + (2/(1 + math.exp(-slope*new_value))) #if disturbance, return this
            new_value = -1 + (2/(1 + math.exp((-slope)*new_value)))
        if activation == 'tanh':
            new_value = np.tanh(new_value)   
        #print ('New value:{}'.format(new_value))
    return new_value


def iter_step (input_row, weight_matrix,activation): 
    
    #print ('Received Row: {}'.format(input_row))
    dc_row = disturbance_row(input_row,weight_matrix)
    dist_matr = dc_row - np.array(input_row)
    #print('Dist Matrix: {}'.format(dist_matr))
    outputs = []
    for attribute_number in range (1,np.size(input_row,0)+1): #for every concept in inputs, calculate every new value
        #print ('--------AT: {}------------'.format(attribute_number))
        new_i = calc_new_value_of_concept (attribute_number, weight_matrix, dist_matr, input_row,activation)
        outputs.append(new_i)
        
    
    outputs = np.array(outputs)
    #print('Returning Raw: {}'.format(outputs))
   
    return outputs


def AFCM(inputs,weight_matrix,patience,max_iterations,activation,slope,verbose):
    
    if verbose:
        print('#####------------STATE SPACE AFCM-------------#######')
        print('#####------no time iterations steps version------#######')
    close = 0
    input_row_1=inputs
    for i in range (max_iterations):
        if verbose:
            print ('STEP: {}'.format(i))
            print('INPUT: {}'.format(input_row_1))
        outputs1 = iter_step (input_row_1, weight_matrix,activation)
        if verbose:
            print('OUTPUT: {}'.format(outputs1))
        diff = abs(input_row_1[len(input_row_1)-1]) - abs(outputs1[len(outputs1)-1])

        if abs(diff)<0.08:
            close = close + 1
        input_row_1 = outputs1
        
        if close>=patience:
            if verbose:
                print('No need for more iterations. System is stabilised')
            stability = 1
            break
        if i == max_iterations-1:
            if verbose:
                print('Reach iterations limit. System is instable')
            stability = 0
            break
        
        print ('--->> Out Value: {}'.format(outputs1[len(outputs1)-1]))
    return outputs1,stability



#%% LOAD WEIGHT MATRIXES FROM THE SAME DIRECORY AS PY
dirname = os.path.dirname(__file__)
def weigth_table_format(number_of_states):
#gets your current directory
    dirname = os.path.dirname(__file__)
    
    #concatenates your current directory with your desired subdirectory
    inputs_path = os.path.join(dirname, r'input.xlsx')
    weight_matrix_path = os.path.join(dirname, r'weigth_matrix.xlsx')
    weight_matrix_state_path = os.path.join(dirname, r'weigth_matrix_state.xlsx')
    weight_matrix_state_out_path = os.path.join(dirname, r'weigth_matrix_state_out.xlsx')
    
    
    #reads the excel file in a dataframe
    weight_matrix = pd.read_excel(weight_matrix_path, header=None)
    weight_matrix_state = pd.read_excel(weight_matrix_state_path, header=None)
    weight_matrix_state_out = pd.read_excel(weight_matrix_state_out_path, header=None)
    inputs =  pd.read_excel(inputs_path, header=None)
    original_ins = inputs
    
    weight_matrix=weight_matrix.fillna(0)
    weight_matrix_state=weight_matrix_state.fillna(0)
    weight_matrix_state_out = weight_matrix_state_out.fillna(0)
    inputs = inputs.fillna(0)
    
    weight_matrix=weight_matrix.replace('VW',0.2)
    weight_matrix=weight_matrix.replace('-VW',-0.2)
    weight_matrix=weight_matrix.replace('W',0.35)
    weight_matrix=weight_matrix.replace('-W',-0.35)
    weight_matrix=weight_matrix.replace('M',0.49)
    weight_matrix=weight_matrix.replace('-M',-0.49)
    weight_matrix=weight_matrix.replace('S',0.75)
    weight_matrix=weight_matrix.replace('-S',-0.75)
    weight_matrix=weight_matrix.replace('VS',0.95)
    weight_matrix=weight_matrix.replace('-VS',-0.95)
    weight_matrix=weight_matrix.astype(float)
    
    weight_matrix_state=weight_matrix_state.replace('VW',0.2)
    weight_matrix_state=weight_matrix_state.replace('-VW',-0.2)
    weight_matrix_state=weight_matrix_state.replace('W',0.35)
    weight_matrix_state=weight_matrix_state.replace('-W',-0.35)
    weight_matrix_state=weight_matrix_state.replace('M',0.49)
    weight_matrix_state=weight_matrix_state.replace('-M',-0.49)
    weight_matrix_state=weight_matrix_state.replace('S',0.75)
    weight_matrix_state=weight_matrix_state.replace('-S',-0.75)
    weight_matrix_state=weight_matrix_state.replace('VS',0.95)
    weight_matrix_state=weight_matrix_state.replace('-VS',-0.95)
    weight_matrix_state=weight_matrix_state.astype(float)
    
    weight_matrix_state_out=weight_matrix_state_out.replace('VW',0.2)
    weight_matrix_state_out=weight_matrix_state_out.replace('-VW',-0.2)
    weight_matrix_state_out=weight_matrix_state_out.replace('W',0.35)
    weight_matrix_state_out=weight_matrix_state_out.replace('-W',-0.35)
    weight_matrix_state_out=weight_matrix_state_out.replace('M',0.49)
    weight_matrix_state_out=weight_matrix_state_out.replace('-M',-0.49)
    weight_matrix_state_out=weight_matrix_state_out.replace('S',0.75)
    weight_matrix_state_out=weight_matrix_state_out.replace('-S',-0.75)
    weight_matrix_state_out=weight_matrix_state_out.replace('VS',0.95)
    weight_matrix_state_out=weight_matrix_state_out.replace('-VS',-0.95)
    weight_matrix_state_out=weight_matrix_state_out.astype(float)    
    
    
    inputs = inputs.drop(inputs.columns[0], axis=1)
    inputs = np.array(inputs).reshape(-1)
    inputs = np.append(inputs,0)
    weight_matrix=np.array(weight_matrix)
    weight_matrix_state=np.array(weight_matrix_state)
    weight_matrix_state_out=np.array(weight_matrix_state_out).reshape(-1,1)
    
    input_num = len(inputs) - 1
    b = np.zeros((input_num,input_num))
    weight_matrix_state_new = np.append(b, weight_matrix_state, axis=1)
    
    return inputs,original_ins,weight_matrix,weight_matrix_state_new, weight_matrix_state_out



def state_initialization(inputs,weight_matrix_state_new,number_of_states,patience,activation,slope,verbose):
    start = weight_matrix_state_new.shape[1] - number_of_states
    end = len(inputs) + number_of_states -1
    
    k = 0
    state_initial = []
    for i in range(start,end):
        print(i)
        if verbose:
            print('Preparing State: {}'.format(k+1))
        weight_matrix_state_temp=np.concatenate((weight_matrix_state_new[:,:len(inputs)-1],weight_matrix_state_new[:,i].reshape(-1,1)),axis=1)
        outs,stability = AFCM(inputs,weight_matrix_state_temp,patience,1,activation,1,verbose)
        
        if verbose:
            print ('Out = {}'.format(outs[len(inputs)-1]))
        
        state_initial.append(outs[len(inputs)-1])
        k=k+1
    return np.array(state_initial).reshape(-1,1)



def afcm_matrix_build(inputs,state_initial,weight_matrix,weight_matrix_state_out,number_of_states,negate_concepts_after_state):
    afcm_inputs = np.concatenate((inputs[:len(inputs)-1].reshape(-1,1),state_initial,inputs[len(inputs)-1].reshape(-1,1)))
    
    w1 = weight_matrix[:,:len(inputs)-1]
    w3 = weight_matrix[:,len(inputs)-1]
    w4 = np.concatenate((w3.reshape(-1,1),weight_matrix_state_out))
    
    c = np.zeros((len(inputs)-1+number_of_states,len(inputs)-1+number_of_states))
    c[:len(inputs)-1,:len(inputs)-1] = w1
    
    
    afcm_weights = np.concatenate((c,w4),axis=1)
    
    for i in range(afcm_weights.shape[0]):
        if i in negate_concepts_after_state:
            afcm_weights[i,afcm_weights.shape[1]-1] = 0

    
    return afcm_inputs,afcm_weights


def rule_conditions(afcm_weights):
    
    if inputs[30] ==1:
        afcm_weights[30,34] = afcm_weights[30,34] + (0.5*afcm_weights[30,34])
    
    if inputs[29] ==1 and inputs[23] ==1:
        afcm_weights[29,34] = afcm_weights[29,34] + (0.2*afcm_weights[29,34])
        afcm_weights[23,34] = afcm_weights[23,34] + (0.2*afcm_weights[23,34])
        
    if inputs[12] ==1:
        if inputs[5] == 0:
            afcm_weights[12,34]=afcm_weights[12,34]+(0.5*afcm_weights[12,34])
    
    if inputs[11] ==1:
        afcm_weights[18,34] =0
    
    if inputs[11] ==0 and inputs[12] ==0 and inputs[19] ==0:
        afcm_weights[21,34] = afcm_weights[21,34] + (0.2*afcm_weights[21,34])
        afcm_weights[23,34] = afcm_weights[23,34] + (0.2*afcm_weights[23,34])
        afcm_weights[25,34] = afcm_weights[25,34] + (0.2*afcm_weights[25,34])
        afcm_weights[27,34] = afcm_weights[27,34] + (0.2*afcm_weights[27,34])
        afcm_weights[29,34] = afcm_weights[29,34] + (0.2*afcm_weights[29,34])
        
    if (inputs[4] ==1 and inputs[30]==1 and (inputs[28]==1 or inputs[26]==1 or inputs[24]==1 or inputs[22]==1)):
        afcm_weights[4,34] = afcm_weights[4,34] - (0.25*afcm_weights[4,34])
        
    return afcm_weights
    
            
        
#%% FCM FUNCTION

activation  = 'sigmoid' #or 'tanh'
slope = 2.5 #experimental sigmoid slope
max_iterations = 10
patience=3
verbose = True
number_of_states = 3

negate_concepts_after_state = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# Load excel tables, return itial tables in numpy array
inputs,original_ins,weight_matrix,weight_matrix_state_new,weight_matrix_state_out = weigth_table_format(number_of_states)

# Calculate the initial values of each state with AFCM with 2 iterations
state_initial = state_initialization(inputs,weight_matrix_state_new,number_of_states,patience,activation,slope,verbose)

# append initial values of state concepts and reform the table of weights to include state-weights. Produces a big weight table, and big input array
afcm_inputs,afcm_weights = afcm_matrix_build(inputs,state_initial,weight_matrix,weight_matrix_state_out,number_of_states,negate_concepts_after_state)

#
afcm_weights = rule_conditions(afcm_weights)



#%% STATE CALCULATOR

outs,stability = AFCM(afcm_inputs.reshape(-1),afcm_weights,patience,max_iterations,activation,slope,verbose)

print ('--->> Out Value: {}'.format(outs[len(outs)-1]))

if outs[len(outs)-1] < -0.4:
    print ('Very Low Possibility')
elif outs[len(outs)-1] < -0.8:
    print ('Low Possibility')
elif outs[len(outs)-1] < 0.2:
    print ('Medium Possibility')
elif outs[len(outs)-1] < 0.6:
    print ('High Possibility')
else:
    print ('Verh High Possibility')
    
    






















