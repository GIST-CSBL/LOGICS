'''
    "chemistry" module with multi-processing capability
'''

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import itertools

from . import global_settings
from .multiprocess_tools import multiprocess_task_on_list, multiprocess_task_many_args

# import same names from tools for having the same functions
from .chemistry import is_valid_smiles, convert_to_canon, get_morganfp_by_smi

# from logics_package.analysis import tansim_to_dist, tansim_to_dist2
# from logics_package.analysis import evaluation_basic
# from logics_package.analysis import calculate_simmat
# from logics_package.analysis import internal_diversity
# from logics_package.analysis import standard_metrics
# from logics_package.analysis import optimal_transport
# from logics_package.analysis import repeated_optimal_transport
# from logics_package.analysis import optimization_metrics

def get_valid_canons(smilist):
    '''
        Get the valid & canonical form of the smiles.
        Please note that different RDKit version could result in different validity for the same SMILES.
    '''
    canons = multiprocess_task_on_list(convert_to_canon, smilist, global_settings.NJOBS_MULTIPROC)
    canons = np.array(canons)
    invalid_ids = np.where(canons==None)[0]
    # insert error string to invalid positions
    canons[invalid_ids] = "<ERR>"

    # Re-checking the parsed smiles, since there are bugs in rdkit parser.
    # https://github.com/rdkit/rdkit/issues/4701
    is_valid = multiprocess_task_on_list(is_valid_smiles, canons, global_settings.NJOBS_MULTIPROC)
    is_valid = np.array(is_valid)
    invalid_ids = np.where(is_valid==False)[0]
    return np.delete(canons, invalid_ids), invalid_ids

def get_fps_from_smilist(smilist, r=2, b=2048):
    """ We assume that all smiles are valid. """
    # zipped input format
    _r = [r for _ in range(len(smilist))]
    _b = [b for _ in range(len(smilist))]
    zipped_input = zip(smilist, _r, _b)
    fps_list = multiprocess_task_many_args(get_morganfp_by_smi, zipped_input, global_settings.NJOBS_MULTIPROC)
    return fps_list

def rdk2npfps(fps_list):
    npfps_list = multiprocess_task_on_list(np.array, fps_list, global_settings.NJOBS_MULTIPROC)
    return np.array(npfps_list)

def to_rdkfp(npfp):
    bitstring="".join(npfp.astype(str))
    return DataStructs.cDataStructs.CreateFromBitString(bitstring)

def np2rdkfps(npfps):
    rdkfps = multiprocess_task_on_list(to_rdkfp, npfps, global_settings.NJOBS_MULTIPROC)
    return rdkfps

def tansim_tuple(tup):
    # tup is a tuple of (fp1, fp2)
    fp1, fp2 = tup[0], tup[1]
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def calculate_simmat(fps1, fps2):
    tuplist = list(itertools.product(fps1, fps2))
    flat_sims = multiprocess_task_on_list(tansim_tuple, tuplist, global_settings.NJOBS_MULTIPROC)
    return np.array(flat_sims).reshape((len(fps1), len(fps2)))
