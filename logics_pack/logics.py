"""
    LOGICS 
    - a framework for Learning optimal Generative distribution Iteratively for the focused Chemical Structures
"""

import numpy as np
import pandas as pd
import pickle as pkl
import json
import torch
from torch.utils.data import DataLoader
from . import smiles_vocab, smiles_lstm, chemistry, analysis

ablation_categories = [None, 'no_memory', 'no_explore', 'no_regular', 'select_determin']

def tournament_selection(pool_size, scores, num_winners):
    """
        random tournament selection (returns the indices)
        should be pool_size > num_winners
    """
    winners = []
    cpool = set(range(pool_size))
    for i in range(num_winners):
        competrs = np.random.choice(list(cpool), 2) # two indices are chosen
        if scores[competrs[0]] > scores[competrs[1]]:
            winners.append(competrs[0])
            cpool.remove(competrs[0])
        else:
            winners.append(competrs[1])
            cpool.remove(competrs[1])
    return winners

class LayeredTournaments:
    """
        Fine-tuning set filtering module using mutiple stages of tournaments

        list_scores should be numpy array of (N, m) where N is the number of 
        tournament layers, and m is the number of the competitors.
        survivor_sizes should be a list of integers, specifying the number of 
        survivors at each tournament layer. It should contain N integers.
    """
    def __init__(self, list_scores: np.array, survivor_sizes):
        self.list_scores = list_scores
        self.layers, self.competitors = list_scores.shape
        self.survivor_sizes = survivor_sizes

    def perform_tournaments(self):
        survivors = np.arange(self.competitors)
        for i in range(self.layers):
            scores = (self.list_scores[i])[survivors]
            winners = tournament_selection(len(scores), scores, self.survivor_sizes[i])
            survivors = survivors[winners]
        return survivors

class SelectDeterministic:
    """
        Fine-tuning set filtering module using deterministic selecction.

        list_scores should be numpy array of (N, m) where N is the number of 
        tournament layers, and m is the number of the competitors.
        survivor_sizes should be a list of integers, specifying the number of 
        survivors at each tournament layer. It should contain N integers.
    """
    def __init__(self, list_scores: np.array, survivor_sizes):
        self.list_scores = list_scores
        self.layers, self.competitors = list_scores.shape
        self.survivor_sizes = survivor_sizes

    def perform_tournaments(self):
        survivors = np.arange(self.competitors)
        for i in range(self.layers):
            scores = (self.list_scores[i])[survivors]
            sorted_inds = np.argsort(scores)[::-1]    # sort ascending order -> descending
            winners = sorted_inds[:self.survivor_sizes[i]]   # filter by deterministic selection
            survivors = survivors[winners]
        return survivors

class ExperienceMemory:
    """
        Memorize the previously found 'good' generated examples.
        This module manages a pandas DataFrame. init_memory_dict is a dictionary
        that initializes the memory DF. One notable feature of this memory is that 
        for updates, it removes the ones that have low scores on priority_column.

        'smiles' is required as a column.
        The official columns used in the paper are ['smiles', 'activity', 'prior NLL'].
    """
    def __init__(self, init_memory_dict, priority_column):
        mem_df = pd.DataFrame(init_memory_dict)
        if 'smiles' not in mem_df.columns:
            print(" 'smiles' key/column is required!")
            self.memory = None
            return
        self.priority_column = priority_column
        # sort by priority (ascending order)
        self.memory = mem_df.sort_values(by=priority_column).reset_index(drop=True)

    def sample(self, ssize):
        mem_inds = np.random.choice(len(self.memory), ssize, replace=False)
        return mem_inds, self.memory.iloc[mem_inds].copy()
    
    def update(self, winners_dict):
        """
            delete individuals that have low scores on priority_column,
            and replace them with the new high-scoring individuals (winners).
            ---- Check if the smiles is not already in the memory. ---- we are not doing this
        """
        winners_df = pd.DataFrame(winners_dict)
        self.memory.iloc[:len(winners_df)] = winners_df
        self.memory.sort_values(by=self.priority_column, inplace=True) # ascending order
        self.memory.reset_index(inplace=True, drop=True)
        return

def LOGICS_training(config):
    """
        Tournament of Bioactivity and Likelihood
        config:
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            pretrained_model_path (.ckpt)
            max_epoch
            save_period
            featurizer (function(smiles) -> feature for predictor)
            predictor_path (.pkl)
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            memory_fmt (.smi with epoch wildcard)
            memory_size
            gen_size
            exp_size
            save_size
            finetune_lr
            finetune_bs
            sampling_bs
    """
    if config.ablation not in ablation_categories:
        print("Ablation category (", config.ablation, ") doesn't exist!")
        print("Please edit the config.ablation in the setting.py file.")
        exit(0)
    elif config.ablation == "no_memory":
        LOGICS_nomem_training(config)
        return

    device_name = config.device_name
    memory_size = config.memory_size
    featurizer = config.featurizer

    vocab = smiles_vocab.Vocabulary()
    vocab.init_from_file(config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab)

    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)

    pret_ckpt = torch.load(config.pretrained_model_path, map_location=device_name)

    lstm_agent = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
    lstm_agent.lstm.load_state_dict(pret_ckpt['model_state_dict'])
    agent_optimizer = torch.optim.Adam(lstm_agent.lstm.parameters(), lr=config.finetune_lr)

    if config.ablation != "no_regular":
        lstm_prior = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
        lstm_prior.lstm.load_state_dict(pret_ckpt['model_state_dict'])
        for param in lstm_prior.lstm.parameters():
            param.requires_grad = False

    with open(config.predictor_path, 'rb') as f:
        pred_model = pkl.load(f)

    # initial memory samples
    ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
    # memory consists of valid canons
    samples = ssplr.sample_clean(memory_size*2, maxlen=150)
    _vacans, _ = chemistry.get_valid_canons(samples)
    mem_init = list(set(_vacans))[:memory_size]

    # record the initial memory prediction values
    mem_features = featurizer(mem_init)
    mem_preds = pred_model.predict(mem_features)
    if config.ablation != "no_regular":
        mem_prior_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(mem_init, lstm_prior, smtk, vocab)
        if len(ke_id_list) > 0:
            print("[-Warning-] The memory contains sequences that cannot be used!!!")
            print(np.array(mem_init)[ke_id_list])
            print("- Aborting the program ...")
            return None

    # initialize the memory
    init_mem_dict = { 'smiles': mem_init, 'activity': mem_preds }
    if config.ablation != "no_regular":
        init_mem_dict['prior_nll'] = mem_prior_nlls
    expmem = ExperienceMemory(init_mem_dict, priority_column='activity')

    # LOGICS training loop
    for epo in range(config.max_epoch+1):
        print("----- epoch: ", epo)
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        samples = ssplr.sample_raw(config.gen_size, maxlen=150)
        samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
        decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)
        
        savings = decoded_samples[:config.save_size]
        # save model and memory
        if epo % config.save_period == 0:
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': lstm_agent.lstm.state_dict(),
                'ft_optimizer_state_dict': agent_optimizer.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in savings])
            expmem.memory.to_csv(config.memory_fmt%epo, index=False)

        # filter valid uniques
        vacans, _ = chemistry.get_valid_canons(decoded_samples)
        vaunis = list(set(vacans))
        if config.ablation != "no_regular":
            vauni_prior_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(vaunis, lstm_prior, smtk, vocab)
            if len(ke_id_list) > 0:
                print("[-Warning-] Generated vacans contain key-error tokens!!")
                print("- These examples will automatically be removed.")
                vaunis = np.delete(vaunis, ke_id_list)
                print("- sanity check passed?: ", len(vaunis)==len(vauni_prior_nlls))
        vauni_features = featurizer(vaunis)
        vauni_preds = pred_model.predict(vauni_features)

        print("-> num uniq: ", len(vaunis)) # debugging
        print("-> avg pred_act: ", vauni_preds.mean())

        # sample experience
        mem_inds, mem_samp = expmem.sample(config.exp_size)
        mem_smis, mem_preds  = mem_samp['smiles'], mem_samp['activity']
        if config.ablation != "no_regular":
            mem_prior_nlls = mem_samp['prior_nll']

        # form the initial competitors
        cmptrs = np.concatenate((vaunis, mem_smis))
        cmptrs_size = len(cmptrs)
        cmptrs_preds = np.concatenate((vauni_preds, mem_preds))
        if config.ablation != "no_regular":
            cmptrs_prior_nlls = np.concatenate((vauni_prior_nlls, mem_prior_nlls))
        if config.ablation != "no_explore":
            cmptrs_agent_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(cmptrs, lstm_agent, smtk, vocab)
            if len(ke_id_list) > 0:
                print("[-Error-] The key error should not happen here ...")
                print("- Aborting the program!!")
                exit(0)

        # we are performing 3 tournament layers ...
        if config.ablation == "no_explore":
            list_scores = np.array([cmptrs_preds, -cmptrs_prior_nlls])
            surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4)]
        elif config.ablation == "no_regular":
            list_scores = np.array([cmptrs_preds, cmptrs_agent_nlls])
            surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4)]
        else:
            list_scores = np.array([cmptrs_preds, cmptrs_agent_nlls, -cmptrs_prior_nlls])
            surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4), int(cmptrs_size/8)]
        
        if config.ablation == "select_determin":
            laytour = SelectDeterministic(list_scores, surv_sizes)
        else:
            laytour = LayeredTournaments(list_scores, surv_sizes)
        survivors = laytour.perform_tournaments() # indices of the final winners

        winners_dict = { 'smiles':cmptrs[survivors], 'activity':cmptrs_preds[survivors] }
        if config.ablation != "no_regular":
            winners_dict['prior_nll'] = cmptrs_prior_nlls[survivors]
        expmem.update(winners_dict)

        # train the agent with the winners
        trainDL = DataLoader(cmptrs[survivors], batch_size=config.finetune_bs)
        epoch_loss = smiles_lstm.perform_train_epoch(smtk, trainDL, lstm_agent, agent_optimizer)

def LOGICS_nomem_training(config):
    print("Training the no-memory case of Ablation ...")
    device_name = config.device_name
    featurizer = config.featurizer

    vocab = smiles_vocab.Vocabulary()
    vocab.init_from_file(config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab)

    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)

    lstm_agent = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
    lstm_prior = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
    pret_ckpt = torch.load(config.pretrained_model_path, map_location=device_name)
    lstm_agent.lstm.load_state_dict(pret_ckpt['model_state_dict'])
    lstm_prior.lstm.load_state_dict(pret_ckpt['model_state_dict'])

    agent_optimizer = torch.optim.Adam(lstm_agent.lstm.parameters(), lr=config.finetune_lr)
    for param in lstm_prior.lstm.parameters():
        param.requires_grad = False

    with open(config.predictor_path, 'rb') as f:
        pred_model = pkl.load(f)

    for epo in range(config.max_epoch+1):
        print("----- epoch: ", epo)
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        samples = ssplr.sample_raw(config.gen_size, maxlen=150)
        samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
        decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)

        savings = decoded_samples[:config.save_size]
        # save model and memory
        if epo % config.save_period == 0:
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': lstm_agent.lstm.state_dict(),
                'ft_optimizer_state_dict': agent_optimizer.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in savings])

        # filter valid uniques
        vacans, _ = chemistry.get_valid_canons(decoded_samples)
        vaunis = list(set(vacans))
        vauni_prior_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(vaunis, lstm_prior, smtk, vocab)
        if len(ke_id_list) > 0:
            print("[-Warning-] Generated vacans contain key-error tokens !!!")
            print("- These examples will automatically be removed.")
            vaunis = np.delete(vaunis, ke_id_list)
            print("- sanity check passed?: ", len(vaunis)==len(vauni_prior_nlls))
        vauni_features = featurizer(vaunis)
        vauni_preds = pred_model.predict(vauni_features)

        print("-> num uniq: ", len(vaunis)) # debugging
        print("-> avg pred_act: ", vauni_preds.mean())

        # form the initial competitors
        cmptrs = np.concatenate((vaunis, []))
        cmptrs_size = len(cmptrs)
        cmptrs_preds = np.concatenate((vauni_preds, []))
        cmptrs_prior_nlls = np.concatenate((vauni_prior_nlls, []))
        cmptrs_agent_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(cmptrs, lstm_agent, smtk, vocab)
        if len(ke_id_list) > 0:
            print("[-Error-] The key error should not happen here ...")
            print("- Aborting the program !!!")
            exit(0)

        # we are performing 3 tournament layers
        list_scores = np.array([cmptrs_preds, cmptrs_agent_nlls, -cmptrs_prior_nlls])
        surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4), int(cmptrs_size/8)]
        laytour = LayeredTournaments(list_scores, surv_sizes)
        survivors = laytour.perform_tournaments() # indices of the final winners

        # train the agent with the winners
        trainDL = DataLoader(cmptrs[survivors], batch_size=config.finetune_bs)
        epoch_loss = smiles_lstm.perform_train_epoch(smtk, trainDL, lstm_agent, agent_optimizer)