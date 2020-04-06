import math
import numpy as np



function  = 'sigmoid' #or 'tanh'

inputs = [[0.8,0.300,0.200,0.750],
          [0.8,0.500,0.500,0.750],
          [0.8,0.200,0.100,0.750]] #experimental input matrix for 2 iteration steps (1st row is the initial state)

w = [[0,  0, 0, 0],
     [0,  0, 0, 0.900],
     [0,  0, 0, 0.900],
     [0,  0, 0, 0]] # wxperimental weight matrix

inputs = np.array(inputs)
w = np.array(w)

#%%
slope = 1.5 #experimental sigmoid slope

#%% Concept value Calculator: give the row of instances before and after, give the weight matrix, give the excact concept t that you refer to
def calc_value (t, w, instance, instancebef): # t is an integer describing which concept we are calculating, w is the weight matrix, instance is the row with the initial values of the concepts
    
    temp = 0
    j = 0
    
    for i in range(0,len(instance)):
        
        #DC: disturbance caused by the change of initial state
        dc = instance - instancebef
        
        #based on dc, calculate the new value of one concept
            
        temp = temp + dc[i]*w[j,t-1]
        j = j+1
    
    sumw = 0 #useless sum of weights (for experiments)
    for i in range (len(w)):
        
        sumw = sumw + abs( w[i, t-1])
        
    ak1 = instance[t-1]+ temp # new value of one concepts
    
    if temp==0:
        ak1n = ak1 # if no disturbance, return null
    else:
        ak1n = 1/(1 + math.exp(-slope*ak1)) #if disturbance, return this
        
    #ak1n = ak1
    #ak3 = math.tanh(ak1)
    #ak2 = 1 / (1 + math.exp(-sumw*ak1))
    return ak1n
    

#%% 

def iter_step (inputs, w, nonew): # nonew: if == 0, then it will calculate the first iteration step (i.e. from row 1 to row 2). If == 1, it will calculate from row 1 to row 2

    instancebef = inputs[nonew]
    instance = inputs [nonew+1]    
    
    for i in range (np.size(inputs,1)): #for every concept in inputs

        new_i = calc_value(i+1, w, instance, instancebef) #calls the calc_value function. the first argument is the concept number for calculation
        instance[i] = new_i # change the instance
        
    return instance

#%%       
                              

allll = iter_step (inputs,w,1)  #test 
