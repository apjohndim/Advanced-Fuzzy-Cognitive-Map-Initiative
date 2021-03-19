# State - Space AFCM initiative
# No time iterations
import math
import numpy as np
from copy import deepcopy

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
    #print('Disturbance: {}'.format(concept_disturbance))
    
    sumw = 0 
    for i in range (len(inputs)-1):
        
        sumw = sumw + abs( weight_matrix[i, attribute_number-1])
    
    #print ('SumW:{}'.format(sumw))
    
    if concept_disturbance==0:
        new_value = inputs[attribute_number-1] # if no disturbance, return null
        #print ('New value: {}'.format(new_value))
    else:
        if sumw>1 or sumw<-1:
            new_value = inputs[attribute_number-1] + (concept_disturbance/sumw)
        else:
            new_value = inputs[attribute_number-1] + concept_disturbance
            
        if activation == 'sigmoid':
            new_value = -1 + (2/(1 + math.exp(-slope*new_value))) #if disturbance, return this
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






def AFCM(inputs,patience,max_iterations,activation,slope,verbose):
    
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
        
        
    return outputs1,stability


#%% INPUTS


inputs = [0.03,   0.046,   0.5,   0.75,   0.50,    0.55] #experimental input matrix for 2 iteration steps (1st row is the initial state)
inputs.append(0) # append the output initial value

weight_matrix = [[0,  0, 0, 0, 0, 0, 0.95],
                 [0,  0, 0, 0, 0, 0, 0.75],
                 [0.5,  0, 0, 0, 0, 0, 0.5],
                 [0,  0, 0, 0, 0, 0, -0.5],
                 [0,  0, 0, 0, 0, 0, -0.75],
                 [0,  0, 0, 0, 0, 0, -0.75]
                                          ]

#y=np.array([np.array(xi) for xi in inputs])
inputs = np.array(inputs, dtype='float64')
weight_matrix = np.array(weight_matrix)




#%% FCM FUNCTION

activation  = 'tanh' #or 'tanh'
slope = 1.5 #experimental sigmoid slope
max_iterations = 20
patience=3
verbose = False
outs,stability = AFCM(inputs,patience,max_iterations,activation,slope,verbose)






















