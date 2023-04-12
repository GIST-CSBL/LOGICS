"""
    This file includes functions calculating metrics for gpc model evaluation.
    Please check each experiment notebooks, and check the subsidiary files needed to
        perform the following evaluations. 
        e.g. *.smi for valid generations and *.npy for fingerprints.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import analysis, chemistry, frechet_chemnet

@dataclass
class EvalConfig:
    ssize : int  # sample size
    vc_smis : list  # valid & canonical smiles generations
    npfps : np.ndarray  # numpy array format of rdkit fingerprint of generations
    simmat_size : int  # size of generations to be used for similarity matrix calculation
    fc_vecs : np.ndarray  # vectors calculated from FreChet ChemNet module
    data_smis : list  # smiles from data, either validation set or test set
    data_rdkfps : list  # data rdkit fingerprints
    data_fc_vecs : np.ndarray  # data vectors
    ot_repeats : int  # how many times OT calculation repeats
       
def eval_standard(evcon:EvalConfig, pret_smis):
    """
        Standard metrics for evaluation
    """
    unis = list(set(evcon.vc_smis))  # unique generations
    pret_set = set(pret_smis)
    novs = list(set(unis).difference(pret_set))
    
    validity = len(evcon.vc_smis) / evcon.ssize
    uniqueness = len(unis) / len(evcon.vc_smis)
    novelty = len(novs) / len(unis)
    
    rdkfps = chemistry.np2rdkfps(evcon.npfps[:evcon.simmat_size])
    intdiv = analysis.internal_diversity(rdkfps)
    return validity, uniqueness, novelty, intdiv

def eval_optimization(evcon:EvalConfig, predictor):
    """
        Optimization metrics for evaluation
    """
    predact = np.mean(predictor.predict(evcon.npfps))
    
    gen_rdkfps = chemistry.np2rdkfps(evcon.npfps)
    ext_simmat = analysis.calculate_simmat(gen_rdkfps[:evcon.simmat_size], evcon.data_rdkfps)
    pwsim = np.mean(ext_simmat)

    fcdval = frechet_chemnet.fcd_calculation(evcon.fc_vecs, evcon.data_fc_vecs)
    
    supply_sz = len(evcon.data_smis) * evcon.ot_repeats
    ot_simmat = analysis.calculate_simmat(gen_rdkfps[:supply_sz], evcon.data_rdkfps)
    ot_distmat = analysis.transport_distmat(analysis.tansim_to_dist, ot_simmat,
                                            num_repeats=evcon.ot_repeats)
    _, _, motds = analysis.repeated_optimal_transport(ot_distmat, evcon.ot_repeats)
    otdval = np.mean(motds)
    return predact, pwsim, fcdval, otdval
