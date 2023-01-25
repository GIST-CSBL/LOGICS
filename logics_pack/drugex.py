"""
    This code is reimplementation of the DrugEx method:
        https://github.com/XuhanLiu/DrugEx

    The original paper of DrugEx:
        Liu, X., Ye, K., van Vlijmen, H.W.T. et al. 
        An exploration strategy improves the diversity of de novo ligands using deep reinforcement learning: a case for the adenosine A2A receptor. 
        J Cheminform 11, 35 (2019). https://doi.org/10.1186/s13321-019-0355-6
"""

import torch
from torch import nn, optim
import copy
import json
import numpy as np
import pickle as pkl
from . import smiles_lstm, smiles_vocab, chemistry, analysis

class DrugEx_Generator(nn.Module):
    """
        The initialization takes our study's generator model as prior and agent.
        Also, the Adam optimizer is defined in here.
    """
    def __init__(self, prior, device, lr=0.0001):
        super(DrugEx_Generator, self).__init__()
        self.agent = copy.deepcopy(prior)
        self.prior = prior
        # deactive the gradient for prior
        for param in self.prior.lstm.parameters():
            param.requires_grad = False

        self.optim = optim.Adam(self.agent.lstm.parameters(), lr=lr)
        self.device = device
    
    # This function returns the samples with epsilon exploration.
    # mutatation is a prior model.
    def evolve(self, batch_size, max_length=150, epsilon=0.01):
        # start_token = Variable(torch.zeros(batch_size).long())
        start_token = torch.zeros(batch_size).long()
        start_token = start_token.to(self.device)
        start_token[:] = self.agent.beg_idx

        x = start_token.view(-1,1) # current step's input token

        hl_units = self.prior.lstm.hidden_layer_units # list sizes of hidden layers
        hidden_states = []
        cell_states = []
        hidden_states2 = [] # exploration (prior) network hidden states
        cell_states2 = []
        for i in range(len(hl_units)):
            # zero_mat = Variable(x.new_zeros(batch_size, hl_units[i]).float())
            zero_mat = x.new_zeros(batch_size, hl_units[i]).float()
            hidden_states.append(zero_mat.clone())
            cell_states.append(zero_mat.clone())
            hidden_states2.append(zero_mat.clone())
            cell_states2.append(zero_mat.clone())

        sequences = []
        finished = torch.zeros(batch_size).byte() # memorize if the example is finished or not.

        for step in range(max_length):
            prob, _, hidden_states, cell_states = self.agent.step_likelihood(x.reshape(batch_size), hidden_states, cell_states)
            prob2, _, hidden_states2, cell_states2 = self.prior.step_likelihood(x.reshape(batch_size), hidden_states2, cell_states2)

            is_mutate = (torch.rand(batch_size) < epsilon).to(self.device) # which to use exploration network
            proba = prob.clone()
            proba[is_mutate, :] = prob2[is_mutate, :]
            
            x = torch.multinomial(proba, num_samples=1).view(-1)
            sequences.append(x.view(-1,1))
            
            x = x.data.clone()
            EOS_sampled = (x == self.agent.eos_idx).data
            finished = torch.ge(finished + EOS_sampled.cpu() , 1)
            if torch.prod(finished) == 1: break 
            
        sequences = torch.cat(sequences, 1)
        return sequences

def REINFORCE(drugex, NLLs, rewards, beta=0.1):
    rewards = torch.Tensor(rewards).to(drugex.device)
    rl_loss = NLLs*(rewards-beta)
    avgloss = rl_loss.mean()
    drugex.optim.zero_grad()
    avgloss.backward()
    drugex.optim.step()
    return rl_loss

def DrugEx_training(config):
    """
        config (=setting from original paper):
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            pretrained_model_path (.ckpt)
            featurizer (function(smiles) -> feature for predictor)
            predictor_path (.pkl)
            max_epoch           
            save_period
            save_size (# sample size for save)
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            train_batch_size (=64)
            scaler (# reward scaler)
            rewarding (one of the reward conversions in reward_functions.py)
            beta (=0.1)
            epsilon (=0.1)
            finetune_lr
            sampling_bs
    """
    device_name = config.device_name
    featurizer = config.featurizer

    vocab = smiles_vocab.Vocabulary()
    vocab.init_from_file(config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab)

    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)

    lstm_prior = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
    pret_ckpt = torch.load(config.pretrained_model_path, map_location=device_name)
    lstm_prior.lstm.load_state_dict(pret_ckpt['model_state_dict'])

    drugex = DrugEx_Generator(lstm_prior, device_name, lr=config.finetune_lr)

    with open(config.predictor_path, 'rb') as f:
        pred_model = pkl.load(f)

    for epo in range(config.max_epoch+1):
        ssplr = analysis.SafeSampler(drugex.agent, config.sampling_bs)
        if epo % config.save_period == 0:
            print('---', epo, '---')
            samples = ssplr.sample_raw(config.save_size, maxlen=150)
            samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
            decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)
            # save model and samples
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': drugex.agent.lstm.state_dict(),
                'ft_optimizer_state_dict': drugex.optim.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in decoded_samples])

        sequences = drugex.evolve(config.train_batch_size, epsilon=config.epsilon, max_length=150)
        sequences = smiles_lstm.truncate_EOS(sequences.detach().cpu().numpy(), vocab)
        train_samples = smiles_lstm.decode_seq_list(sequences, vocab)
        # valid and canonical
        train_vacans, invids = chemistry.get_valid_canons(train_samples)
        vids = np.delete(np.arange(config.train_batch_size), invids) # valid indices
        # excape batch when there are no valid examples
        if len(vids) <= 0:
            print("No valid samples detected!")
            continue

        # perform prediction with valid samples
        vc_features = featurizer(train_vacans)
        vc_preds = pred_model.predict(vc_features)
        # convert activity to reward
        vc_rewards = config.rewarding(vc_preds)
        # reward = -0.5 for invalid smiles

        # reward assignment to each sample
        train_rewards = np.full(config.train_batch_size, -0.5)
        train_rewards[vids] = vc_rewards

        # scaled rewards
        train_rewards = train_rewards*config.scaler

        ### debug ###
        if epo % int(config.save_period/3) == 0:
            print("valid count: ", len(vids))
            print("avg pkx: ", vc_preds.mean())

        # encode samples
        enc_samples, keyerror_ids = smiles_lstm.prepare_batch(train_samples, smtk, vocab)
        # if key error occurs, it can not be learned.
        if len(keyerror_ids) > 0:
            train_rewards = np.delete(train_rewards, keyerror_ids)
        nlls, _ = drugex.agent.likelihood(enc_samples)
        # REINFORCE agent update
        rl_loss = REINFORCE(drugex, nlls, train_rewards, beta=config.beta)
        