import numpy as np
import pandas as pd
import math
import logging
from adam_optimizer import AdamOptimizer
from pandas.api.types import is_numeric_dtype
logging.basicConfig(level=logging.ERROR)
from dataset_processing_scratch import DatasetProcess, DatasetError

#_________Adam Optimizer Parameters & Instance_____________________

ith = 1
hyperParameter1 = 0.9000
hyperParameter2 = 0.9900
smoothening = 1e-8
learningRate =0.0005

adam_intc = AdamOptimizer()
adam_optimizer = adam_intc.adam_optimizer
# _______ Other Initialization___________________

totalLoss = 0
neuronWeights = np.random.randn(16,14) * np.sqrt(2/14)
neuronWeights_hiddenLayer_1 = np.random.randn(8,16) * np.sqrt(2/16)
neuronWeights_Final_Output = np.random.randn(1, 8)* np.sqrt(2/8)

neuronBias = np.zeros(16)
neuronBias_hiddenLayer_1 = np.zeros(8)
neuronBias_Final_Output = np.zeros(1)

batch_gradient_last_layer = np.zeros_like(neuronWeights_Final_Output)
batch_gradient_middle_layer = np.zeros_like(neuronWeights_hiddenLayer_1)
batch_gradient_input_layer = np.zeros_like(neuronWeights)
batch_gradient_bias_middle_layer=np.zeros_like(neuronBias_hiddenLayer_1)
batch_gradient_bias_input_layer = np.zeros_like(neuronBias )
gradient_last_layer_l_y = np.zeros_like(neuronBias_Final_Output)

inputLayerDerivative = np.random.randn(16)
hiddenLayerOneOutput = np.array([0]*16)

lastLayer_firstMomentVector = np.zeros_like(neuronWeights_Final_Output)
lastLayer_secondMomentVector = np.zeros_like(neuronWeights_Final_Output)
lastLayerBias_firstMomentVector = np.zeros_like(neuronBias_Final_Output)
lastLayerBias_secondMomentVector = np.zeros_like(neuronBias_Final_Output)

middleLayer_firstMomentVector = np.zeros_like(neuronWeights_hiddenLayer_1)
middleLayer_secondMomentVector = np.zeros_like(neuronWeights_hiddenLayer_1)
middleLayerBias_firstMomentVector = np.zeros_like(neuronBias_hiddenLayer_1)
middleLayerBias_secondMomentVector = np.zeros_like(neuronBias_hiddenLayer_1)

inputLayer_firstMomentVector = np.zeros_like(neuronWeights)
inputLayer_secondMomentVector = np.zeros_like(neuronWeights)
inputLayerBias_firstMomentVector = np.zeros_like(neuronBias)
inputLayerBias_secondMomentVector = np.zeros_like(neuronBias)

        

# fullPath = "/home/ebuka/Downloads/sales.csv"
fullPath = "/home/ebuka/deeplearning/housing-predictor-from-scratch/Housing.csv"
try:
    data_class = DatasetProcess(fullPath,"price" ,0.7) 
    X_train, y_train , X_test, y_test,column_names = data_class.transform_columns()
except DatasetError as e:
    print({e})

# ___________ forward propagation_________________________________

def inputLayerOne(eachRow):
    global inputLayerOutput
    global neuronWeights
    global neuronBias
    global inputLayerDerivative
    
    x = np.array(eachRow)             # shape: (14,)
    W = np.array(neuronWeights)       # shape: (16, 14)
    b = np.array(neuronBias)          # shape: (16,)
    z = W @ x + b                     # matrix multiplication + bias shape: (16,)
    
    inputLayerDerivative = (z > 0).astype(float)
    inputLayerOutput = np.maximum(0, z)
    return inputLayerOutput


def hiddenLayer1(neuronOutputOne):
    global  hiddenLayerOneOutput
    global  neuronWeights_hiddenLayer_1
    global  neuronBias_hiddenLayer_1
    global hiddenLayerOneDerivative

    
    x = np.array(neuronOutputOne)                # shape: (16,)
    W = np.array(neuronWeights_hiddenLayer_1)    # shape: (8, 16)
    b = np.array(neuronBias_hiddenLayer_1)       # shape: (8,)

    
    z = W @ x + b
    
    hiddenLayerOneDerivative = (z > 0).astype(float)
    hiddenLayerOneOutput = np.maximum(0, z) 
    return hiddenLayerOneOutput                 # shape: (8,)
    
def outputLayer(neuronOutputFinal):
    global finalLayerOutput
    global neuronWeights_Final_Output
    global neuronBias_Final_Output
    
    x = np.array(neuronOutputFinal)            # shape: (8,)
    W = np.array(neuronWeights_Final_Output)
    b = np.array(neuronBias_Final_Output)         # shape: (1,8)

    z = W @ x + b
    
    # return z
    return z.item()

def sumTarget(y_target, y_pred):
    return np.mean((y_target - y_pred) ** 2)

def lastLayerDecent(y_pred, y_train,X):
    global neuronWeights_Final_Output, neuronBias_Final_Output, hiddenLayerOneOutput
    global batch_gradient_last_layer, gradient_last_layer_l_y
    global lastLayer_firstMomentVector, lastLayer_secondMomentVector,lastLayerBias_firstMomentVector , lastLayerBias_secondMomentVector, ith
    global batch_counter
    
    oldWeight = neuronWeights_Final_Output.copy()
    change_L_Y = np.array([2 * (y_pred - y_train)])
    change_L_W =  np.outer(change_L_Y,hiddenLayerOneOutput )    # shape (8,)

# summing gradients for mini-batch
    gradient_last_layer_l_y +=change_L_Y 
    batch_gradient_last_layer += change_L_W
    if batch_counter > 0 and batch_counter % 32 == 0:  
        change_L_W = batch_gradient_last_layer /32
        change_L_Y = gradient_last_layer_l_y /32
        
        batch_gradient_last_layer = np.zeros_like(neuronWeights_Final_Output)
        gradient_last_layer_l_y = np.zeros_like(neuronBias_Final_Output)

        change_L_W,neuronWeights_Final_Output,ith,lastLayer_firstMomentVector, lastLayer_secondMomentVector
        
        adam_weight_dict = {'grad': change_L_W,'oldWeightMatrix':neuronWeights_Final_Output,'ith':ith,
                           'lastLayer_firstMoment':lastLayer_firstMomentVector,
                           'lastLayer_secondMoment':lastLayer_secondMomentVector,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}
       
        adam_bias_dict = {'grad': change_L_Y,'oldWeightMatrix':neuronBias_Final_Output,'ith':ith,
                           'lastLayer_firstMoment':lastLayerBias_firstMomentVector,
                           'lastLayer_secondMoment':lastLayerBias_secondMomentVector,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}
  
        neuronWeights_Final_Output, lastLayer_firstMomentVector, lastLayer_secondMomentVector  = adam_optimizer(adam_weight_dict)
        neuronBias_Final_Output , lastLayerBias_firstMomentVector,  lastLayerBias_secondMomentVector = adam_optimizer(adam_bias_dict  )
        
        # neuronWeights_Final_Output -= 0.001 * change_L_W 
        # neuronBias_Final_Output -= 0.001 * change_L_Y
        
        
# Called the middleLayerDescent function
    middleLayerDescent(change_L_Y,X,oldWeight)
       

def middleLayerDescent(change_L_Y,X,oldWeight):
    global neuronWeights_Final_Output , neuronWeights_hiddenLayer_1, inputLayerOutput
    global neuronBias_hiddenLayer_1
    global batch_gradient_middle_layer, batch_gradient_bias_middle_layer, batch_counter
    global middleLayer_firstMomentVector, middleLayer_secondMomentVector,middleLayerBias_firstMomentVector , middleLayerBias_secondMomentVector, ith

    oldHiddenWeights = neuronWeights_hiddenLayer_1.copy()
    delta2 =    oldWeight.T @ change_L_Y 
    delta2 = delta2 * hiddenLayerOneDerivative
    grad = np.outer(delta2 , inputLayerOutput)

# summing gradients for mini-batch
    batch_gradient_middle_layer += grad
    batch_gradient_bias_middle_layer +=delta2
    
    if batch_counter > 0 and batch_counter % 32 == 0:
        grad = batch_gradient_middle_layer / 32
        delta2 = batch_gradient_bias_middle_layer / 32
        batch_gradient_middle_layer = np.zeros_like(neuronWeights_hiddenLayer_1)
        batch_gradient_bias_middle_layer=np.zeros_like(neuronBias_hiddenLayer_1)
    
        # neuronWeights_hiddenLayer_1  -= 0.001 * grad
        # neuronBias_hiddenLayer_1 -= 0.001 * delta2

        adam_weight_dict = {'grad': grad,'oldWeightMatrix':neuronWeights_hiddenLayer_1,'ith':ith,
                           'lastLayer_firstMoment':middleLayer_firstMomentVector,
                           'lastLayer_secondMoment':middleLayer_secondMomentVector,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}
        adam_bias_dict = {'grad': delta2,'oldWeightMatrix':neuronBias_hiddenLayer_1,'ith':ith,
                           'lastLayer_firstMoment':middleLayerBias_firstMomentVector,
                           'lastLayer_secondMoment':middleLayerBias_secondMomentVector,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}

        neuronWeights_hiddenLayer_1, middleLayer_firstMomentVector, middleLayer_secondMomentVector  = adam_optimizer(adam_weight_dict  )
        neuronBias_hiddenLayer_1 , middleLayerBias_firstMomentVector,  middleLayerBias_secondMomentVector = adam_optimizer( adam_bias_dict )
        

    firstLayerDescent(delta2,X,oldHiddenWeights)

    # return neuronWeights_hiddenLayer_1 

def firstLayerDescent(delta2, inputX_j,oldWeight):
    global neuronWeights
    global neuronBias
    global inputLayerDerivative
    global batch_gradient_input_layer, batch_gradient_bias_input_layer, batch_counter
    global inputLayer_firstMomentVector, inputLayer_secondMomentVector, inputLayerBias_firstMomentVector , inputLayerBias_secondMomentVector, ith
    
    inputX_j = np.array(inputX_j)
    delta1 = oldWeight.T @ delta2
    delta1 = delta1 * inputLayerDerivative
    grad1 = np.outer(delta1, inputX_j)
    
# summing gradients for mini-batch
    batch_gradient_input_layer += grad1 
    batch_gradient_bias_input_layer += delta1
    
    if batch_counter > 0 and batch_counter % 32 == 0:

        grad1 = batch_gradient_input_layer / 32
        delta1 = batch_gradient_bias_input_layer / 32
        batch_gradient_input_layer = np.zeros_like(neuronWeights)
        batch_gradient_bias_input_layer = np.zeros_like(neuronBias )
        # neuronWeights -= 0.001 * grad1
        # neuronBias -= 0.001 * delta1

        adam_weight_dict = {'grad': grad1,'oldWeightMatrix':neuronWeights,'ith':ith,
                           'lastLayer_firstMoment':inputLayer_firstMomentVector,
                           'lastLayer_secondMoment':inputLayer_secondMomentVector ,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}
        adam_bias_dict = {'grad': delta1,'oldWeightMatrix':neuronBias,'ith':ith,
                           'lastLayer_firstMoment':inputLayerBias_firstMomentVector,
                           'lastLayer_secondMoment':inputLayerBias_secondMomentVector,'hyperParameter1':hyperParameter1,
                           'hyperParameter2':hyperParameter2,'smoothening':smoothening,'learningRate':learningRate}
  
        neuronWeights, inputLayer_firstMomentVector, inputLayer_secondMomentVector  = adam_optimizer(adam_weight_dict )
        neuronBias , inputLayerBias_firstMomentVector,  inputLayerBias_secondMomentVector = adam_optimizer(adam_bias_dict )

#_________This increments the ADAM Optimzer Timer__________________________________________________
        ith +=1 

for epoch in range(1000):
    totalLoss = 0
    batch_counter=0
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    for i in range(len(y_train)):
        batch_counter +=1
        y_pred = outputLayer(hiddenLayer1(inputLayerOne(X_train[i])))
        totalLoss += sumTarget(y_train[i], y_pred)
        
        lastLayerDecent( y_pred,y_train[i],X_train[i])

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", totalLoss / len(y_train))







