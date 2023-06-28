"""
    This code is directly from:
        https://github.com/schwallergroup/augmented_memory

    The code was slightly modified to fit in our project.
"""

# -------------------------------------------------------------------------------------------------------------
# this file has been modified from https://github.com/MolecularAI/Reinvent for Augmented Memory implementation
# -------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Tuple, List

from .inception_configuration import InceptionConfiguration
from .conversions import Conversions
from copy import deepcopy
###
from . import smiles_lstm, smiles_vocab

class PriorWrapper:
    ### Note: "likelihood" stored in memory is actually the prior's NLL.
    def __init__(self, prior_generator: smiles_lstm.SmilesLSTMGenerator, vocab: smiles_vocab.Vocabulary, 
                 smtk: smiles_vocab.SmilesTokenizer):
        self.prior_generator = prior_generator
        self.vocab = vocab
        self.smtk = smtk

    def likelihood_smiles(self, smiles: List[str]):
        enc_samples, _ = smiles_lstm.prepare_batch(smiles, self.smtk, self.vocab)
        prior_nlls, _ = self.prior_generator.likelihood(enc_samples)
        return prior_nlls  # dim: (batch_size), type: Tensor

class ScoringFunc:
    def __init__(self, smi_to_score: dict):
        self.smi_to_score = smi_to_score

    def get_final_score(self, smiles: List[str]):
        return np.array([self.smi_to_score[smi] for smi in smiles])

class Inception:
    ### Note: "likelihood" stored in memory is actually the prior's NLL.
    ### def __init__(self, configuration: InceptionConfiguration, scoring_function, prior):
    def __init__(self, configuration: InceptionConfiguration, scoring_function: ScoringFunc, prior: PriorWrapper):
        self.configuration = configuration
        self._chemistry = Conversions()
        self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'score', 'likelihood'])
        self._load_to_memory(scoring_function, prior, self.configuration.smiles)

    def _load_to_memory(self, scoring_function, prior, smiles):
        if len(smiles):
            standardized_and_nulls = [self._chemistry.convert_to_rdkit_smiles(smile) for smile in smiles]
            standardized = [smile for smile in standardized_and_nulls if smile is not None]
            self.evaluate_and_add(standardized, scoring_function, prior)

    def _purge_memory(self):
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values('score', ascending=False)
        self.memory = sorted_df.head(self.configuration.memory_size)
        self.memory = self.memory.loc[self.memory['score'] != 0.0]

    def mode_collapse_guard(self):
        # in *pure* exploitation scenarios where Selective Memory Purge is not used, the following heuristic
        # pre-emptively guards against rare cases of mode collapse at suboptimal values
        sliced_memory = self.memory.head(int(self.configuration.memory_size*0.5))
        if (sliced_memory['score'].nunique() == 1) and (int(sliced_memory['score'].iloc[0]) != 1):
            print("---- Pre-emptively guarding against mode collapse: purging buffer -----")
            self.memory = pd.DataFrame(columns=['smiles', 'score', 'likelihood'])

    def selective_memory_purge(self, smiles, score):
        zero_score_indices = np.where(score == 0.)[0]
        if len(zero_score_indices) > 0:
            smiles_to_purge = smiles[zero_score_indices]
            scaffolds_to_purge = [self._chemistry.get_scaffold(smiles) for smiles in smiles_to_purge]
            purged_memory = deepcopy(self.memory)
            purged_memory['scaffolds'] = purged_memory['smiles'].apply(self._chemistry.get_scaffold)
            purged_memory = purged_memory.loc[~purged_memory['scaffolds'].isin(scaffolds_to_purge)]
            purged_memory.drop('scaffolds', axis=1, inplace=True)
            self.memory = purged_memory
        else:
            return

    def evaluate_and_add(self, smiles, scoring_function, prior):
        if len(smiles) > 0:
            score = scoring_function.get_final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            ### df = pd.DataFrame({"smiles": smiles, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            df = pd.DataFrame({"smiles": smiles, "score": score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self._purge_memory()

    def add(self, smiles, score, neg_likelihood):
        # NOTE: likelihood should be already negative
        df = pd.DataFrame({"smiles": smiles, "score": score, "likelihood": neg_likelihood.detach().cpu().numpy()})
        self.memory = pd.concat([self.memory, df])
        self._purge_memory()

    def sample(self) -> Tuple[List[str], np.array, np.array]:
        sample_size = min(len(self.memory), self.configuration.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], []

    def augmented_memory_replay(self, prior) -> Tuple[List[str], np.array, np.array]:
        if len(self.memory) != 0:
            smiles = self.memory["smiles"].values
            # randomize the smiles
            ### randomized_smiles_list = self._chemistry.get_randomized_smiles(smiles, prior)
            randomized_smiles_list = [self._chemistry.randomize_smiles(smi) for smi in smiles]
            scores = self.memory["score"].values
            prior_likelihood = -prior.likelihood_smiles(randomized_smiles_list).cpu()
            return randomized_smiles_list, scores, prior_likelihood
        else:
            return [], [], []


