"""
    This module would NOT directly be imported to tbl_packages.__init__
    due to the external dependency to fcd.
"""

import fcd
import numpy as np

def chemnet_matrix_save(smi_fmt, chnt_fmt, epochs):
    """
        Gets smi_fmt (.smi with epoch wildcard), 
        writes files with ChemNet matrix from FCD package (.npy).
        epochs should be a list of integers.
    """
    chnt_model = fcd.load_ref_model()
    for epo in epochs:
        print(epo)
        with open(smi_fmt%epo, 'r') as f:
            vacans = [line.strip() for line in f.readlines()]
        chnt_mat = fcd.get_predictions(chnt_model, vacans)
        np.save(chnt_fmt%epo, chnt_mat)

def fcd_calculation(chnt1, chnt2):
    """
        Calculate Frechet ChemNet Distance between two sets of ChemNet vectors.
    """
    mu1, sigma1 = np.mean(chnt1, axis=0), np.cov(chnt1.T)
    mu2, sigma2 = np.mean(chnt2, axis=0), np.cov(chnt2.T)
    return fcd.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def fcd_list_save(ua_chnt, gen_chnt_fmt, epochs, fcd_list_path):
    fcd_list = []
    for epo in epochs:
        gen_chnt = np.load(gen_chnt_fmt%epo)
        fcdval = fcd_calculation(ua_chnt, gen_chnt)
        fcd_list.append(fcdval)
    np.save(fcd_list_path, fcd_list)