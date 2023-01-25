"""
    Fine-tuning method by simulating drug-synthesis-test cycles from:
        Marwin H. S. Segler, Thierry Kogej, Christian Tyrchan, and Mark P. Waller
        ACS Central Science 2018 4 (1), 120-131
        DOI: 10.1021/acscentsci.7b00512
        https://pubs.acs.org/doi/10.1021/acscentsci.7b00512
"""

import torch
from torch.utils.data import DataLoader
import json
import numpy as np
import pickle as pkl
from . import smiles_lstm, smiles_vocab, chemistry, analysis

def Segler_training(config):
    """
        config (=setting from original paper):
            device_name
            tokens_path (.txt)
            pretrain_setting_path (.json)
            pretrained_model_path (.ckpt)
            featurizer (function(smiles) -> feature for predictor)
            predictor_path (.pkl)
            score_thrs (# molecule is active when score > score_thrs)
            max_epoch (=40)
            save_period
            save_size (# sample size for save)
            save_ckpt_fmt (.ckpt with epoch wildcard)
            sample_fmt (.txt with epoch wildcard)
            init_gen_size (=100000)
            ssz_per_epoch (=10000 # sample size per epoch)
            ft_period (=5 # fine-tune period)
            finetune_lr
            finetune_bs
            sampling_bs
            record_actives_size (# max size for recorded actives)
            record_actives_fmt (.smi with epoch wildcard)
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

    # initial set of active molecules (record_actives)
    ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
    samples = ssplr.sample_clean(config.init_gen_size, maxlen=150)
    vacans, _ = chemistry.get_valid_canons(samples)
    # unique generations
    uniqs = list(set(vacans))
    un_features = featurizer(uniqs)
    un_preds = pred_model.predict(un_features)
    act_ids = np.where(un_preds > config.score_thrs)[0]
    record_actives = (np.array(uniqs)[act_ids]).tolist()
    rec_act_preds = un_preds[act_ids]

    record_samples = []
    for epo in range(config.max_epoch+1):
        print('---', epo, '---')
        if epo % config.save_period == 0:
            ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
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
            # save record_actives
            with open(config.record_actives_fmt%epo, 'w') as f:
                f.writelines([line+'\n' for line in record_actives])

        # fine-tune at each epoch
        trainDL = DataLoader(record_actives, batch_size=config.finetune_bs)
        epoch_loss = smiles_lstm.perform_train_epoch(smtk, trainDL, lstm_agent, agent_optimizer)

        # sampling at each epoch
        ssplr = analysis.SafeSampler(lstm_agent, config.sampling_bs)
        samples = ssplr.sample_clean(config.ssz_per_epoch, maxlen=150)
        record_samples.extend(samples)
        if (epo+1) % config.ft_period == 0:
            # time to update record_actives
            vacans, _ = chemistry.get_valid_canons(record_samples)
            uniqs = list(set(vacans))
            # only bring in the ones not in the record
            uniqs = set(uniqs).difference(set(record_actives))
            if len(uniqs) == 0:
                print(">> No exclusively unique generations are detected!! <<")
                continue
            uniqs = list(uniqs)
            un_features = featurizer(uniqs)
            un_preds = pred_model.predict(un_features)
            #### for debug ####
            print("-- average prediction: ", un_preds.mean())
            ###################
            act_ids = np.where(un_preds > config.score_thrs)[0]
            if len(act_ids) == 0:
                print(">> There are no active molecules in the new generations!! <<")
                continue
            record_actives.extend((np.array(uniqs)[act_ids]).tolist())
            rec_act_preds = np.concatenate((rec_act_preds, un_preds[act_ids]))
            record_samples = []

            # if the record size is over than record_actives_size, then drop the low-activity molecules
            if len(record_actives) > config.record_actives_size:
                print(">> record overflow! <<")
                sorted_ids = np.argsort(rec_act_preds) # ascending
                sorted_ids = sorted_ids[::-1] # descending
                remain_ids = sorted_ids[:config.record_actives_size]
                record_actives = np.array(record_actives)[remain_ids].tolist()
                rec_act_preds = rec_act_preds[remain_ids]

