"""
    This file contains:
    1. pre-training of SMILES generator (get prior model when finished)
    2. Vanilla GPC model (VGPC) fine-tuning phase
    3. standard fine-tuning method: fine-tune the generator with fixed data
"""

import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pickle as pkl
from . import smiles_lstm, smiles_vocab, chemistry, analysis

def pretrain(config):
    """
        config:
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            dataset_path (.smi)
            max_epoch
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            sample_size
    """
    vocab = smiles_vocab.Vocabulary()
    vocab.init_from_file(config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab)

    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)

    # initial model
    lstm_generator = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], config.device_name)
    optimizer = torch.optim.Adam(lstm_generator.lstm.parameters(), lr = model_setting['initial_lr'])
    
    # dataset
    with open(config.dataset_path, 'r') as f:
        trainset = [line.strip() for line in f.readlines()]
    
    train_data_loader = DataLoader(trainset, batch_size=model_setting['batch_size'],
                              shuffle=True, drop_last=True, collate_fn=None)

    for epoch in range(1, config.max_epoch+1):
        print("-------- epoch: ", epoch)
        if epoch >= 3:
            ckpt_old_old = torch.load(config.save_ckpt_fmt%(epoch-2))
            NLL1 = ckpt_old_old['last_epoch_loss']
            ckpt_old = torch.load(config.save_ckpt_fmt%(epoch-1))
            NLL2 = ckpt_old['last_epoch_loss']
            # check the NLL decrease amount of samples1 -> samples2
            if (NLL1 - NLL2) < model_setting['NLL_dec_thr']:
                # decrease learning rate if the condition is met.
                print("Decreasing the learning rate because NLL decrease was not enough!!!")
                epoch_lr = ckpt_old['last_epoch_lr'] * model_setting['lr_decay']
            else:
                # if we observed enough increase, keep the learning rate.
                epoch_lr = ckpt_old['last_epoch_lr']        
        else:
            # if we have only done 0 or 1 epochs, ...
            epoch_lr = model_setting['initial_lr']
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = epoch_lr
        print("learning rate: ", optimizer.param_groups[0]['lr'])
        
        epoch_loss = smiles_lstm.perform_train_epoch(smtk, train_data_loader, lstm_generator, optimizer, verbose=True)
        print("epoch loss: ", epoch_loss)
        
        # sample and check the validity
        print("start sampling ...")
        ssplr = analysis.SafeSampler(lstm_generator, model_setting['batch_size'])
        samples = ssplr.sample_raw(config.sample_size, maxlen=150)
        samples = samples.detach().cpu().numpy()
        samples = smiles_lstm.truncate_EOS(samples, vocab)
        decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)

        print("saving the samples ...")
        # save the samples
        with open(config.sample_fmt%(epoch), 'w') as f:
            f.writelines([line+'\n' for line in decoded_samples])
        
        # check the validity
        vacans, invids = chemistry.get_valid_canons(decoded_samples)
        # empty generation is not considered as valid
        _vacans = []
        for smi in vacans:
            if len(smi) > 0:
                _vacans.append(smi)
        epoch_valid = len(_vacans) / config.sample_size
        print("epoch validity: ", epoch_valid)
        
        ckpt_dict = {
            'initial_lr': model_setting['initial_lr'], 'lr_decay': model_setting['lr_decay'],
            'batch_size': model_setting['batch_size'], 'emb_size': model_setting['emb_size'],
            'hidden_units': model_setting['hidden_units'], 'epochs_done': epoch,
            'last_epoch_loss': epoch_loss, 'last_epoch_lr': epoch_lr, 'last_epoch_valid': epoch_valid,
            
            'model_state_dict': lstm_generator.lstm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(ckpt_dict, config.save_ckpt_fmt%(epoch))

def VanillaGPC_training(config):
    """
        Vanilla (the most basic version) of GPC transfer
        config:
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            pretrained_model_path (.ckpt)
            featurizer (function(smiles) -> feature for predictor)
            predictor_path (.pkl)
            max_epoch
            save_period
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            gen_size
            high_score_size
            finetune_lr
            finetune_bs
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

    # fine-tuning
    for epo in range(config.max_epoch+1):
        print("----- epoch: ", epo)
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        samples = ssplr.sample_raw(config.gen_size, maxlen=150)
        samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
        decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)

        # save model
        if epo % config.save_period == 0:
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': lstm_agent.lstm.state_dict(),
                'ft_optimizer_state_dict': agent_optimizer.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in decoded_samples])

        # filter valid uniques
        vacans, _ = chemistry.get_valid_canons(decoded_samples)
        vaunis = list(set(vacans))
        vauni_features = featurizer(vaunis)
        vauni_preds = pred_model.predict(vauni_features)

        print("-> num uniq: ", len(vaunis)) # debugging
        print("-> avg pred_act: ", vauni_preds.mean())

        asc_scr_idx = np.argsort(vauni_preds) # sort by score (ascending order)
        des_scr_idx = asc_scr_idx[::-1]
        top_scr_idx = des_scr_idx[:config.high_score_size]
        top_scr_smis = np.array(vaunis)[top_scr_idx].tolist()

        # train the agent with the top score molecules
        trainDL = DataLoader(top_scr_smis, batch_size=config.finetune_bs)
        epoch_loss = smiles_lstm.perform_train_epoch(smtk, trainDL, lstm_agent, agent_optimizer)

def finetune_standard(config, ft_dataset):
    """
        Standard fine-tuning for transfer learning
        ft_dataset should be the smiles list.
        config:
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            pretrained_model_path (.ckpt)
            max_epoch
            save_period
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            gen_size
            finetune_lr
            finetune_bs
            sampling_bs
    """
    device_name = config.device_name

    vocab = smiles_vocab.Vocabulary()
    vocab.init_from_file(config.tokens_path)
    smtk = smiles_vocab.SmilesTokenizer(vocab)

    with open(config.pretrain_setting_path, 'r') as f:
        model_setting = json.load(f)

    lstm_agent = smiles_lstm.SmilesLSTMGenerator(vocab, model_setting['emb_size'], model_setting['hidden_units'], device_name)
    pret_ckpt = torch.load(config.pretrained_model_path, map_location=device_name)
    lstm_agent.lstm.load_state_dict(pret_ckpt['model_state_dict'])

    agent_optimizer = torch.optim.Adam(lstm_agent.lstm.parameters(), lr=config.finetune_lr)

    # fine-tuning
    for epo in range(config.max_epoch+1):
        print("----- epoch: ", epo)
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        samples = ssplr.sample_raw(config.gen_size, maxlen=150)
        samples = smiles_lstm.truncate_EOS(samples.detach().cpu().numpy(), vocab)
        decoded_samples = smiles_lstm.decode_seq_list(samples, vocab)

        # save model
        if epo % config.save_period == 0:
            ckpt_dict = {
                'ft_setup': None, 'emb_size': model_setting['emb_size'], 'hidden_units': model_setting['hidden_units'],
                'epoch': epo, 'model_state_dict': lstm_agent.lstm.state_dict(),
                'ft_optimizer_state_dict': agent_optimizer.state_dict()
            }
            torch.save(ckpt_dict, config.save_ckpt_fmt%epo)
            with open(config.sample_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in decoded_samples])

            # train the agent with the dataset
            trainDL = DataLoader(ft_dataset, batch_size=config.finetune_bs)
            epoch_loss = smiles_lstm.perform_train_epoch(smtk, trainDL, lstm_agent, agent_optimizer)
