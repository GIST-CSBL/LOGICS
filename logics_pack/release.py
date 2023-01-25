"""
    This code is reimplementation of the ReLeaSE method:
        https://github.com/isayev/ReLeaSE

    The original paper of ReLeaSE:
        Mariya Popova, Olexandr Isayev, Alexander Tropsha. 
        Deep Reinforcement Learning for de-novo Drug Design. 
        Science Advances, 2018, Vol. 4, no. 7, eaap7885. DOI: 10.1126/sciadv.aap7885

    This file also includes the ReLeaSE+ model, which incorporates LOGICS tournament and memory
    mechanisms with the original ReLeaSE to boost exploration.
"""

import torch
import json
import numpy as np
import pickle as pkl
import pandas as pd
from . import smiles_lstm, smiles_vocab, chemistry, analysis
from . import logics

def reinforce_decaying_rewards(encoded_samples, rewards, gamma, lstm_agent, device_name):
    """
        This function returns the loss function of ReLeaSE method.
    """
    # one hot matrix of non-<PAD> positions
    non_paddings = (~(encoded_samples == lstm_agent.pad_idx)).to(device_name)

    # position likelihoods seq_likes (batch_size, max_len_of_batch)
    _, seq_likes = lstm_agent.likelihood(encoded_samples)
    _, maxlen = seq_likes.shape

    # REINFORCE with decaying reward
    discounted_rewards = torch.Tensor(rewards).to(device_name) # initial reward
    rl_loss_batch = torch.zeros(len(rewards)).to(device_name) # NLL recording
    # from time 0 to n
    for t in range(maxlen):
        like_step = seq_likes[:,t]
        id_nonpad = torch.where(non_paddings[:,t] == True)
        # update only the positions with non-<PAD> tokens,
        nonpad_loss = torch.log(like_step[id_nonpad])*discounted_rewards[id_nonpad]
        rl_loss_batch[id_nonpad] -= nonpad_loss
        discounted_rewards = discounted_rewards * gamma
    avgloss = rl_loss_batch.mean()
    return avgloss

def ReLeaSE_training(config):
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
            train_batch_size (=15)
            scaler (# reward scaler)
            rewarding (one of the reward conversions in reward_functions.py)
            gamma (=0.97)
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
    pret_ckpt = torch.load(config.pretrained_model_path, map_location=device_name)
    lstm_agent.lstm.load_state_dict(pret_ckpt['model_state_dict'])

    agent_optimizer = torch.optim.Adam(lstm_agent.lstm.parameters(), lr=config.finetune_lr)

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
        train_vacans, invids = chemistry.get_valid_canons(train_samples) # get valid canonicals and invalid ids
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
            print("uniq valid count: ", len(vids))
            print("avg pkx: ", vc_preds.mean())

        # encode samples
        enc_samples, _ = smiles_lstm.prepare_batch(train_samples, smtk, vocab)
        # REINFORCE with decaying rewards
        avgloss = reinforce_decaying_rewards(enc_samples, train_rewards, config.gamma, lstm_agent, device_name)
        agent_optimizer.zero_grad()
        avgloss.backward()
        agent_optimizer.step()

def ReLeaSE_plus_training(config):
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

            memory_fmt
            memory_size
            gen_size
            exp_size

            scaler (# reward scaler)
            rewarding (one of the reward conversions in reward_functions.py)
            gamma (=0.97)
            finetune_lr
            sampling_bs
    """    
    device_name = config.device_name
    featurizer = config.featurizer
    memory_size = config.memory_size

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

    # initial memory samples
    ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
    # memory consists of valid canons
    samples = ssplr.sample_clean(memory_size*2, maxlen=150)
    _vacans, _ = chemistry.get_valid_canons(samples)
    mem_init = list(set(_vacans))[:memory_size]

    # record the initial memory prediction values
    mem_features = featurizer(mem_init)
    mem_preds = pred_model.predict(mem_features)
    mem_prior_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(mem_init, lstm_prior, smtk, vocab)
    if len(ke_id_list) > 0:
        print("[-Warning-] The memory contains sequences that cannot be used!!!")
        print(np.array(mem_init)[ke_id_list])
        print("- Aborting the program ...")
        return None

    # initialize the memory
    init_mem_dict = {
        'smiles':mem_init, 'activity':mem_preds, 'prior_nll': mem_prior_nlls
    }
    expmem = logics.ExperienceMemory(init_mem_dict, priority_column='activity')
    
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
            expmem.memory.to_csv(config.memory_fmt%epo)

        gen_samples = ssplr.sample_clean(config.gen_size, maxlen=150)
        gen_vacans, invids = chemistry.get_valid_canons(gen_samples) # get valid canonicals and invalid ids
        inv_samples = np.array(gen_samples)[invids]

        vaunis = list(set(gen_vacans))
        vauni_prior_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(vaunis, lstm_prior, smtk, vocab)
        if len(ke_id_list) > 0:
            print("[-Warning-] Generated vacans contain key-error tokens !!!")
            print("- These examples will automatically be removed.")
            vaunis = np.delete(vaunis, ke_id_list)
            print("- sanity check passed?: ", len(vaunis)==len(vauni_prior_nlls))
        vauni_features = featurizer(vaunis)
        vauni_preds = pred_model.predict(vauni_features)

        if epo % int(config.save_period/3) == 0:    
            print("-> num uniq: ", len(vaunis)) # debugging
            print("-> avg pred_act: ", vauni_preds.mean())

        # sample experience
        mem_inds, mem_samp = expmem.sample(config.exp_size)
        mem_smis, mem_preds, mem_prior_nlls = mem_samp['smiles'], mem_samp['activity'], mem_samp['prior_nll']

        # form the initial competitors
        cmptrs = np.concatenate((vaunis, mem_smis))
        cmptrs_size = len(cmptrs)
        cmptrs_preds = np.concatenate((vauni_preds, mem_preds))
        cmptrs_prior_nlls = np.concatenate((vauni_prior_nlls, mem_prior_nlls))
        cmptrs_agent_nlls, ke_id_list = smiles_lstm.get_NLLs_batch(cmptrs, lstm_agent, smtk, vocab)
        if len(ke_id_list) > 0:
            print("[-Error-] The key error should not happen here ...")
            print("- Aborting the program !!!")

        # we are performing 3 tournament layers
        list_scores = np.array([cmptrs_preds, cmptrs_agent_nlls, -cmptrs_prior_nlls])
        surv_sizes = [int(cmptrs_size/2), int(cmptrs_size/4), int(cmptrs_size/8)]
        laytour = logics.LayeredTournaments(list_scores, surv_sizes)
        survivors = laytour.perform_tournaments() # indices of the final winners

        win_smis = cmptrs[survivors]
        win_acts = cmptrs_preds[survivors]
        winners_dict = {
            'smiles':win_smis, 'activity':win_acts, 'prior_nll':cmptrs_prior_nlls[survivors]
        }
        expmem.update(winners_dict)

        train_smis = np.concatenate((win_smis, inv_samples))
        train_rewards = np.full(len(train_smis), -0.5) # -0.5 for invalid smiles
        train_rewards[:len(win_smis)] = config.rewarding(win_acts)
        # scaled rewards
        train_rewards = train_rewards*config.scaler

        # encode samples
        enc_samples, _ = smiles_lstm.prepare_batch(train_smis, smtk, vocab)
        # REINFORCE with decaying rewards
        avgloss = reinforce_decaying_rewards(enc_samples, train_rewards, config.gamma, lstm_agent, device_name)
        agent_optimizer.zero_grad()
        avgloss.backward()
        agent_optimizer.step()
