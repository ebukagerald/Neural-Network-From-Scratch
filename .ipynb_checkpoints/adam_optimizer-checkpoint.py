# _________________ADAM OPTIMIZER___________________1,134,34,56,90,1,9,8
                
class AdamOptimizer
        
    def adam_optimizer(grad,oldWeightMatrix,ith,lastLayer_firstMoment,lastLayer_secondMoment):
    
        m1 = hyperParameter1 * lastLayer_firstMoment + (1 - hyperParameter1)*grad
        v1 = hyperParameter2 * lastLayer_secondMoment+ (1 - hyperParameter2 )*grad**2
    
        firstMomentBias = m1/(1-hyperParameter1**ith)
        secondMomentBias = v1/(1-hyperParameter2**ith)
        
        newWeight = oldWeightMatrix - learningRate * firstMomentBias / (np.sqrt(secondMomentBias) + smoothening) 
        return newWeight , m1, v1

