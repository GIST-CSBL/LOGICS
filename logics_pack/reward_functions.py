
import numpy as np

def pAff_to_reward_t1(pAff):
    """
        type 1
        pAff can be the bioactivity measures e.g. pIC, pKx, pCHEMBL. 
        It should be in numpy 1-d array.
    """
    return np.tanh((pAff-6.2)*0.3)

def pAff_to_reward_t2(pAff):
    """
        type 2
        pAff can be the bioactivity measures e.g. pIC, pKx, pCHEMBL. 
        It should be in numpy 1-d array.
    """
    return np.tanh((pAff-6.8)*0.3)

class AffinityRewardTransform:
    def __init__(self, alpha=0.3, beta=6.5):
        self.alpha = alpha
        self.beta = beta
    
    def pAff_to_reward(self, pAff):
        return np.tanh((pAff-self.beta)*self.alpha)