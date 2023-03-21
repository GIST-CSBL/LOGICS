"""
    This module would NOT directly be imported to __init__
    due to the external dependency to fcd.
"""

import fcd
import numpy as np

def fcd_calculation(chnt1, chnt2):
    """
        Calculate Frechet ChemNet Distance between two sets of ChemNet vectors.
    """
    mu1, sigma1 = np.mean(chnt1, axis=0), np.cov(chnt1.T)
    mu2, sigma2 = np.mean(chnt2, axis=0), np.cov(chnt2.T)
    return fcd.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
