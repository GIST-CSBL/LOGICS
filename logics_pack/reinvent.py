"""
    This code is reimplementation of the REINVENT method:
        https://github.com/MarcusOlivecrona/REINVENT

    The original paper of REINVENT:
        Olivecrona, M., Blaschke, T., Engkvist, O. et al. 
        Molecular de-novo design through deep reinforcement learning. 
        J Cheminform 9, 48 (2017). https://doi.org/10.1186/s13321-017-0235-x
"""

import torch
import json
import numpy as np
import pickle as pkl
from . import smiles_lstm, smiles_vocab, chemistry, analysis

def REINVENT_training(config):
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
            sigma (=7)
            rewarding (one of the reward conversions in reward_functions.py)
            train_batch_size
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
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        if epo % config.save_period == 0:
            print('---', epo, '---')
            samples = ssplr.sample_raw(config.save_size, maxlen=150)
            samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
            decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)
            # save model and samples
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': lstm_agent.lstm.state_dict(),
                'ft_optimizer_state_dict': agent_optimizer.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in decoded_samples])

        train_samples = ssplr.sample_clean(config.train_batch_size, maxlen=150)
        train_samples = list(set(train_samples)) # remove duplicates
        new_bs = len(train_samples) # num of uniques
        train_vacans, invids = chemistry.get_valid_canons(train_samples) # get valid canonicals and invalid ids
        vids = np.delete(np.arange(new_bs), invids) # valid indices
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
        train_rewards = np.full(new_bs, -0.5)
        train_rewards[vids] = vc_rewards
        # advantage
        advantages = config.sigma * train_rewards
        
        ### debug ###
        if epo % int(config.save_period/3) == 0:
            print("uniq valid count: ", len(vids))
            print("avg pkx: ", vc_preds.mean())

        # encode samples
        enc_samples, _ = smiles_lstm.prepare_batch(train_samples, smtk, vocab)

        # calculate log likelihood of prior and agent
        prior_nlls, _ = lstm_prior.likelihood(enc_samples)
        agent_nlls, _ = lstm_agent.likelihood(enc_samples)
        prior_loglikes = -prior_nlls
        agent_loglikes = -agent_nlls

        # augmented log likelihood
        augme_loglikes = prior_loglikes + torch.Tensor(advantages).to(device_name)
        # REINVENT loss
        loss = torch.pow((augme_loglikes - agent_loglikes), 2)
        avgloss = loss.mean()

        agent_optimizer.zero_grad()
        avgloss.backward()
        agent_optimizer.step()

        