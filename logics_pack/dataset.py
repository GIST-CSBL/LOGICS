"""
    The ChEMBL dataset pre-built for GuacaMol was downloaded from:
        https://figshare.com/articles/dataset/GuacaMol_All_SMILES/7322252

    code reference and KOR dataset source:
        https://github.com/larngroup/DiverseDRL
"""

import numpy as np
import csv
from . import chemistry, smiles_vocab, pubchem_tools

def find_salts_undefined_tokens(smiles, smtk: smiles_vocab.SmilesTokenizer):
    """
        This function returns the indices that have salts and undefined tokens in the vocab.
    """
    exclude = []
    for i, smi in enumerate(smiles):
        if '.' in smi:
            exclude.append(i)
            continue
        try:
            tokens = smtk.tokenize(smi)
            _ = smtk.vocab_obj.encode(tokens)
        except KeyError as err:
            exclude.append(i)
    return exclude

class PubChemProcessLOGICS(pubchem_tools.PubChemAssaysEntrezGene):
    def __init__(self, entrezid=None):
        super().__init__(entrezid)
    
    def filter_del_undefined_tokens(self, smtk: smiles_vocab.SmilesTokenizer):
        exclude = []
        for i, smi in enumerate(self.filtered['smiles']):
            try:
                tokens = smtk.tokenize(smi)
                _ = smtk.vocab_obj.encode(tokens)
            except KeyError as err:
                exclude.append(i)
        self.filtered = self.filtered.drop(exclude).reset_index(drop=True)
        print("following indices of records are dropped due to undefined tokens:")
        print(exclude)

def process_chembl(guacamol_chembl_path, smtk: smiles_vocab.SmilesTokenizer):
    """ 
        guacamol_chembl_path: smiles file from ...
            https://figshare.com/articles/dataset/GuacaMol_All_SMILES/7322252 
            guacamol_v1_all.smiles

        This function will re-filter the molecules with chemistry.get_valid_canons(),
        and then remove the duplicate SMILES from the original.
        Then, remove examples that contain undefined tokens for our generator models.
    """
    with open(guacamol_chembl_path, 'r') as f:
        guacamol_chembl = [line.strip() for line in f.readlines()]
    new_chembl, _ = chemistry.get_valid_canons(guacamol_chembl)
    exclude_ids = find_salts_undefined_tokens(new_chembl, smtk)
    new_chembl = np.delete(np.array(new_chembl), exclude_ids).tolist()
    # delete duplicates
    new_chembl = list(set(new_chembl))
    return new_chembl

def process_DiverseDRL_KOR(kor_raw_path):
    """ KOR original data: data_clean_kop.csv """
    idx_smiles = 0
    idx_labels = 1
    raw_smiles = []
    raw_labels = []
    with open(kor_raw_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        it = iter(reader)
        next(it, None)  # skip first item.    
        for row in it:
            try:
                raw_smiles.append(row[idx_smiles])
                raw_labels.append(float(row[idx_labels]))
            except:
                pass
    
    smiles = []
    labels = []
    for i in range(len(raw_smiles)):
        if 'a' not in raw_smiles[i] and 'Z' not in raw_smiles[i] and 'K' not in raw_smiles[i]:
            smiles.append(raw_smiles[i])
            labels.append(raw_labels[i])

    return smiles, labels

def fold_splits(data_size, folds):
    fold_size = int(data_size/folds)
    ids = list(range(data_size))
    np.random.shuffle(ids)
    fold_dict = dict()
    for i in range(folds-1):
        fold_members = ids[i*fold_size:(i+1)*fold_size]
        fold_dict[i] = fold_members
    fold_dict[folds-1] = ids[(folds-1)*fold_size:]
    return fold_dict
