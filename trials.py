import math
import numpy as np



function  = 'sigmoid' #or 'tanh'

inputs = [[0.8,0.300,0.200,0.750],
          [0.8,0.500,0.500,0.750],
          [0.8,0.200,0.100,0.750]] #size: batch, conc1, conc2, cunc3

w = [[0,  0, 0, 0],
     [0,  0, 0, 0.900],
     [0,  0, 0, 0.900],
     [0,  0, 0, 0]] # size: 3x3

inputs = np.array(inputs)
w = np.array(w)

#%%
# we assume that we have 3 concepts. Γυρίζει τη νέα τιμή του comcept mono. 
#t=1
#instancebef = inputs[0]  
#instance = inputs [1]

slope = 1.5

#%% Concept value Calculator: give the row of instances before and after, give the weight matrix, give the excact concept t that you refer to
def calc_value (t, w, instance, instancebef): # t is an integer describing which concept we are calculating, w is the weight matrix, instance is the row with the initial values of the concepts
    
    temp = 0
    j = 0
    
    for i in range(0,len(instance)):
        
        #ορισε τον DC matrix
        dc = instance - instancebef
        
        #can calc με τις equations gia dc matrix
            
        temp = temp + dc[i]*w[j,t-1]
        j = j+1
    
    sumw = 0
    for i in range (len(w)):
        
        sumw = sumw + abs( w[i, t-1])
        
    ak1 = instance[t-1]+ temp
    
    if temp==0:
        ak1n = ak1
    else:
        ak1n = 1/(1 + math.exp(-slope*ak1))
        
    #ak1n = ak1
    #ak3 = math.tanh(ak1)
    #ak2 = 1 / (1 + math.exp(-sumw*ak1))
    return ak1n
    

#%% Τωρα πρέπει να υπολογίσεις με τη σειρά την νέα τιμή που παίρνει κάθε cocepts, και με βάση αυτή να υπολογιστεί η αλλαγή στο επόμενο! όχι με 
#βαση την αρχική. 

def iter_step (inputs, w, nonew):

    instancebef = inputs[nonew]
    instance = inputs [nonew+1]    
    
    for i in range (np.size(inputs,1)):
        

        new_i = calc_value(i+1, w, instance, instancebef)
        instance[i] = new_i
        
    return instance

#%%       
                              

allll = iter_step (inputs,w,1)