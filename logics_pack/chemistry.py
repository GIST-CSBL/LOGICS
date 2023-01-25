from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from . import sascorer

def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol == None: return False
    return True

def convert_to_canon(smi, verbose=None):
    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        if verbose: print('[ERROR] cannot parse: ', smi)
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

def get_valid_canons(smilist):
    '''
        Get the valid & canonical form of the smiles.
        Please note that different RDKit version could result in different validity for the same SMILES.
    '''
    canons = []
    invalid_ids = []
    for i, smi in enumerate(smilist):
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            invalid_ids.append(i)
            canons.append(None)
        else:
            canons.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    # Re-checking the parsed smiles, since there are bugs in rdkit parser.
    # https://github.com/rdkit/rdkit/issues/4701
    re_canons = []
    for i, smi in enumerate(canons):
        if smi == None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol == None:
            print("rdkit bug occurred!!")
            invalid_ids.append(i)
        else:
            re_canons.append(smi)
    return re_canons, invalid_ids

def get_morganfp_by_smi(smi, r=2, b=2048):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, nBits=b)
    return fp

def get_fps_from_smilist(smilist, r=2, b=2048):
    """ We assume that all smiles are valid. """
    fps = []
    for i, smi in enumerate(smilist):
        fps.append(get_morganfp_by_smi(smi, r, b))
    return fps

def rdk2npfps(fps_list):
    """ fps_list: list of MorganFingerprint objects """
    return np.array(fps_list)

def np2rdkfps(npfps):
    rdkfps = []
    for npfp in npfps:
        bitstring="".join(npfp.astype(str))
        rdkfp = DataStructs.cDataStructs.CreateFromBitString(bitstring)
        rdkfps.append(rdkfp)
    return rdkfps

# molecular weights MW
def get_MWs(mols):
    return [Descriptors.ExactMolWt(mol) for mol in mols]

# QED
def get_QEDs(mols):
    return [Chem.QED.qed(mol) for mol in mols]

# SAS
def get_SASs(mols):
    return [sascorer.calculateScore(mol) for mol in mols]

# logP
def get_logPs(mols):
    return [Descriptors.MolLogP(mol) for mol in mols]

# TPSA
def get_TPSAs(mols):
    return [Descriptors.TPSA(mol) for mol in mols]

# Murcko Scaffold, returning canonical SMILES form
def get_MrkScfs(mols):
    scaf_mols = [MurckoScaffold.GetScaffoldForMol(mol) for mol in mols]
    scaf_smis = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in scaf_mols]
    return scaf_smis