"""
    This file includes:
    1. SafeSampler class which helps the smiles_lstm to sample molecules
    2. functions for generation analysis
"""

import numpy as np
from typing import List
from rdkit import DataStructs
from scipy import optimize
import torch
from . import smiles_lstm, smiles_vocab

# conversion of Tanimoto similarity to the distance
tansim_to_dist = lambda ts: np.power(10, 1-ts) - 1

class SafeSampler:
    """
    Generating the molecules at once will consumes a lot of memory.
    This tool solves the problem by generating a small amount at a time.
    """
    def __init__(self, model: smiles_lstm.SmilesLSTMGenerator, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def sample_raw(self, number_samples: int, maxlen: int) -> torch.Tensor:
        """ The output format is the integer(token index) matrix. (number_samples x maxlen)"""
        sampled_count = 0
        # fill with <PAD>
        generation = torch.full((number_samples, maxlen), self.model.pad_idx, dtype=torch.long).to(self.model.device_name) 
        while sampled_count < number_samples:
            tokens_mat, _ = self.model.sample(self.batch_size, maxlen)
            bs, slen = tokens_mat.shape
            if (sampled_count+bs) >= number_samples:
                generation[sampled_count:number_samples, :slen] = tokens_mat[:number_samples-sampled_count]
            else:
                generation[sampled_count:(sampled_count+bs), :slen] = tokens_mat
            sampled_count += bs
        return generation[:number_samples]

    def sample_clean(self, number_samples: int, maxlen: int) -> List[str]:
        """ The output only contains the sequences that don't have special tokens. """
        generation = []
        while len(generation) <= number_samples:
            tokens_list, _ = self.model.sample(self.batch_size, maxlen)
            EOS_exist = [] # store which sample includes EOS token
            for i in range(self.batch_size):
                if self.model.eos_idx in tokens_list[i]:
                    EOS_exist.append(i)
            tokens_have_EOS = tokens_list[EOS_exist,:]

            # cut off after the first <EOS>
            trunc_seq_list = smiles_lstm.truncate_EOS(tokens_have_EOS, self.model.voc)
            # check if each seq has a special token ...
            # "clean" == not having special tokens
            clean_seqs = []
            for seq in trunc_seq_list:
                spinds = smiles_vocab.locate_specials(self.model.voc, seq)
                if len(spinds) == 0:
                    clean_seqs.append(seq)

            smiles_list = smiles_lstm.decode_seq_list(clean_seqs, self.model.voc) 
            generation.extend(smiles_list)
        return generation[:number_samples]

def calculate_simmat(fps1, fps2):
    """ Calculate the similarity matrix between two fingerprint lists. """
    simmat = np.zeros((len(fps1), len(fps2)))
    for i in range(len(fps1)):
        for j in range(len(fps2)):
            simmat[i,j] = DataStructs.FingerprintSimilarity(fps1[i], fps2[j])
    return simmat

def internal_diversity(fps):
    """ fps: list of morgan fingerprints """
    simmat = calculate_simmat(fps, fps)
    return (1-simmat).mean()

def optimal_transport(distmat):
    """ 
        Given distmat (row:generation, col:unseen active),
        compute the optimal transport mapping, 
        and return the mapping and tOTD (total optimal transport distance).
    """
    row_ind, col_ind = optimize.linear_sum_assignment(distmat)
    totd = distmat[row_ind, col_ind].sum()
    return row_ind, col_ind, totd

def repeated_optimal_transport(distmat, repeat):
    """
        distmat consists of ((repeat*supply) x demand) distances.
        Optimal transport is calculated with each repeat (certain generation set),
        and the result of mappings is saved as two nested lists: row_ind_nest, col_ind_nest.
        row_ind_nest[i] and col_ind_nest[i] are the OT mapping at the i-th (supply x demand) distmat.
        totds[i] is the total optimal transport distance of the i-th OTD repeat.
    """
    ssize, dsize = distmat.shape
    if dsize*repeat > ssize:
        print("supply size is smaller than (repeat x demand size)!!!")
        print(">>> Abort from optimal transport calculation")
        return None
    row_ind_nest, col_ind_nest, totds = [], [], []
    for i in range(repeat):
        sub_distmat = distmat[i*dsize:(i+1)*dsize]
        ri, ci, totd = optimal_transport(sub_distmat)
        row_ind_nest.append(ri)
        col_ind_nest.append(ci)
        totds.append(totd)
    return row_ind_nest, col_ind_nest, totds

def transport_distmat(ts_to_dist, simmat:np.array, num_repeats=1):
    """
        Given Tanimoto simmat (row:supply(gen), column:demand(data)),
        calculate the transport matrix with specified number of transport repeats.
        Returns matrix with size ((num_repeats*column_size) x column_size)
    """
    rows, cols = simmat.shape  # provided supply and demand size
    # if available supply amount lacks compare to the demand*repeat, abort.
    if rows > cols*num_repeats:
        print("Supply size is smaller than (repeat x demand size) in simmat!")
        print("Aborting...")
        return None
    supplies, demands = cols*num_repeats, cols
    _simmat = simmat[:supplies,:demands]
    return ts_to_dist(_simmat)
