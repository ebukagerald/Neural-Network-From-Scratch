import numpy as np
# _________________ADAM OPTIMIZER___________________1,134,34,56,90,1,9,8
                
class AdamOptimizer:
    
    
    
    def adam_optimizer(self,adam_opt_dict):
    
        grad = adam_opt_dict['grad']
        oldWeightMatrix = adam_opt_dict['oldWeightMatrix']
        ith = adam_opt_dict['ith']
        lastLayer_firstMoment = adam_opt_dict['lastLayer_firstMoment']
        lastLayer_secondMoment = adam_opt_dict['lastLayer_secondMoment']
        hyperParameter1 = adam_opt_dict['hyperParameter1']
        hyperParameter2 = adam_opt_dict['hyperParameter2']
        smoothening = adam_opt_dict['smoothening']
        learningRate = adam_opt_dict['learningRate']
        
    
        m1 = hyperParameter1 * lastLayer_firstMoment + (1 - hyperParameter1)*grad
        v1 = hyperParameter2 * lastLayer_secondMoment+ (1 - hyperParameter2 )*grad**2
    
        firstMomentBias = m1/(1-hyperParameter1**ith)
        secondMomentBias = v1/(1-hyperParameter2**ith)
        
        newWeight = oldWeightMatrix - learningRate * firstMomentBias / (np.sqrt(secondMomentBias) + smoothening) 
        return newWeight , m1, v1

