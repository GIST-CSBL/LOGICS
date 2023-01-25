
"""
The implementation of generative LSTM is hugely influenced by REINVENT,
https://github.com/MarcusOlivecrona/REINVENT

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List

from .smiles_vocab import SmilesTokenizer

# The use of collate_fn is for (<PAD>)-padding. 
# The arr is expected to be an encoded smiles array;
# that is, each element is float.
def collate_fn(arr, PAD_idx):
    """Function to take a list of encoded sequences and turn them into a batch"""
    max_length = max([seq.size for seq in arr])
    collated_arr = torch.full((len(arr), max_length), PAD_idx, dtype=torch.float32)
    for i, seq in enumerate(arr):
        collated_arr[i, :seq.size] = torch.Tensor(seq)
    return collated_arr

def prepare_batch(smiles_list, tokenizer, vocab_obj):
    """ 
        Get a batch of SMILES, turn them into a training batch.
        Also, return the index of the smiles that raised KeyError during encoding.
        Thus, the size of smiles_list is not always equal to sample_batch,
        since error-occurring smiles are removed.
    """
    EOS_idx = vocab_obj.get_EOS_idx()
    PAD_idx = vocab_obj.get_PAD_idx()
    sample_batch_t = [tokenizer.tokenize(smiles) for smiles in smiles_list]
    sample_batch_e = []
    keyerror_ids = []
    for i, tokens in enumerate(sample_batch_t):
        try:
            encoded = vocab_obj.encode(tokens)
        except KeyError as err:
            keyerror_ids.append(i)
            print("KeyError at %s"%smiles_list[i])
            continue
        sample_batch_e.append(encoded)
    # add <EOS> at the end
    EOS_batch = []
    for tokens in sample_batch_e:
        tokens = list(tokens)
        tokens.append(EOS_idx)
        EOS_batch.append(np.array(tokens, dtype=np.float64))
    # pad each example to the length of the longest in the batch
    sample_batch = collate_fn(EOS_batch, PAD_idx)
    return sample_batch, keyerror_ids

class MultiLSTM(nn.Module):
    """ Implements a multi layer LSTM cell including an embedding layer
        and an output linear layer back to the size of the vocabulary """
    def __init__(self, emb_size, hidden_layer_units: List, voc_size):
        if len(hidden_layer_units) < 1:
            raise "There should be at least one hidden layer!"
        super(MultiLSTM, self).__init__()
        self.embedding = nn.Embedding(voc_size, emb_size)
        self.hidden_layer_units = hidden_layer_units
        self.num_hidden_layers = len(hidden_layer_units)
        self.lstm_list = nn.ModuleList()
        self.lstm_list.append(nn.LSTMCell(emb_size, hidden_layer_units[0]))
        for i in range(1, len(hidden_layer_units)):
            self.lstm_list.append(nn.LSTMCell(hidden_layer_units[i-1], hidden_layer_units[i]))
        self.linear = nn.Linear(hidden_layer_units[self.num_hidden_layers-1], voc_size)

        self.voc_size = voc_size
        self.emb_size = emb_size

    # forward() call performs only one step through the timeline.
    # Though, the process is done on the batch-wise.
    # x.shape = (batch_size) <- each example's t-th step token index
    # hs[i].shape = (batch_size, hl_units[i])
    def forward(self, x, hs: List, cs: List):
        emb_x = self.embedding(x)
        # emb_x.shape = (batch_size, feature_dim)
        hs[0], cs[0] = self.lstm_list[0](emb_x, (hs[0], cs[0]))
        for i in range(1, len(hs)):
            hs[i], cs[i] = self.lstm_list[i](hs[i-1], (hs[i], cs[i]))
        fc_out = self.linear(hs[len(hs)-1])
        return fc_out, hs, cs

class SmilesLSTMGenerator():
    """ LSTM language model for generating synthetic SMILES strings """
    def __init__(self, voc, emb_size, hidden_layer_units: List, device_name="cuda"):
        self.device_name = device_name
        self.lstm = MultiLSTM(emb_size, hidden_layer_units, voc.vocab_size)
        self.lstm.to(device_name)
        self.voc = voc
        self.beg_idx = voc.get_BEG_idx()
        self.eos_idx = voc.get_EOS_idx()
        self.pad_idx = voc.get_PAD_idx()

    def step_likelihood(self, xi, hidden_states, cell_states):
        """
            This function returns the prob and log_prob of xi given states.
            Args:
                xi: (batch_size)
        """
        # Note that we are not using x[:, step].view(-1,1) here.
        # If you look at the forward() of MultiLSTM, you see that it expects (batch_size).
        logits, hidden_states, cell_states = self.lstm(xi, hidden_states, cell_states)

        # logits.shape = (batch_size, vocab_size)
        log_prob = nn.functional.log_softmax(logits, dim=1)
        prob = nn.functional.softmax(logits, dim=1)
        return prob, log_prob, hidden_states, cell_states
        
    def likelihood(self, target):
        """
            For given examples of a batch, return the probability for each example.
            Retrieves the likelihood (NLL) of a given sequence,
            it will be used for training LSTM.

            Args:
                target: (batch_size, seq_len) A batch of sequences in integer. 
                <EOS> and <PAD> should be already in. <BEG> is not in.

            Outputs:
                NLLosses: (batch_size) negative log likelihood for each example
                likelihoods: (batch_size, seq_length) likelihood for each position at each example
        """
        target = torch.Tensor(target.float())
        target = target.to(self.device_name).long()
        # It is expected that all the seqs end with at least one EOS.
        # When making the input x, we will cut that last token at the end,
        # and add BEG token at the begining.
        batch_size, seq_length = target.size()
        start_token = target.new_zeros(batch_size, 1).long()
        start_token[:] = self.beg_idx # initialize with all BEG

        hl_units = self.lstm.hidden_layer_units
        hidden_states = []
        cell_states = []
        for i in range(len(hl_units)):
            hidden_state = target.new_zeros(batch_size, hl_units[i]).float() 
            hidden_states.append(hidden_state)
            cell_state = target.new_zeros(batch_size, hl_units[i]).float() 
            cell_states.append(cell_state)

        x = torch.cat((start_token, target[:, :-1]), 1)    
        NLLosses = target.new_zeros(batch_size).float() 
        likelihoods = target.new_zeros(batch_size, seq_length).float()
        for step in range(seq_length):
            # Note that we are sliding a vertical scanner (height=batch_size) moving on timeline.      
            x_step = x[:, step] # (batch_size)
            # let's find x_t[i] where it is <PAD>. Only <PAD>s will be True.
            padding_where = (x_step == self.pad_idx)
            # padding_where.shape = (batch_size)
            non_paddings = ~padding_where

            prob, log_prob, hidden_states, cell_states = self.step_likelihood(x_step, hidden_states, cell_states)

            # the output of the lstm should be compared to the ones at x_step+1 (=target_step)
            # not the exactly same x_step!!
            one_hot_labels = nn.functional.one_hot(target[:, step], num_classes=self.voc.vocab_size)

            # one_hot_labels.shape = (batch_size, vocab_size)
            # Make all the <PAD> tokens as zero vectors.
            one_hot_labels = one_hot_labels * non_paddings.reshape(-1,1)

            likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1)
            loss = one_hot_labels * log_prob
            loss_on_batch = -torch.sum(loss, 1) # this is the negative log loss
            NLLosses += loss_on_batch

        return NLLosses, likelihoods
  
    def sample(self, batch_size, max_length=150):
        """
            Sample a batch of sequences, this will be used for sampling SMILES.
            The returning sequence is in rectangular shape.
            That is, some examples will contain junk tokens after the <EOS> positions.

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
                seqs: (batch_size, seq_length) The sampled sequences.
                likelihoods: 
                log_probs : (batch_size) Log likelihood for each sequence.
        """
        start_token = torch.zeros(batch_size).long() 
        start_token = start_token.to(self.device_name)
        start_token[:] = self.beg_idx

        x = start_token.view(-1,1)
        # x.shape == (batch_size, 1)

        hl_units = self.lstm.hidden_layer_units
        hidden_states = []
        cell_states = []
        for i in range(len(hl_units)):
            hidden_state = x.new_zeros(batch_size, hl_units[i]).float() 
            hidden_states.append(hidden_state)
            cell_state = x.new_zeros(batch_size, hl_units[i]).float() 
            cell_states.append(cell_state)

        sequences = []
        likelihoods = x.new_zeros(batch_size, max_length).float() 
        finished = torch.zeros(batch_size).byte() # memorize if the example is finished or not.

        # x starts as a start_token.
        for step in range(max_length):
            prob, log_prob, hidden_states, cell_states = self.step_likelihood(x.reshape(batch_size), hidden_states, cell_states)
            # prob.shape = (batch_size, vocab_size)

            # After this multinomial sampling, x is set to a sampled token.
            # x = torch.multinomial(prob).view(-1)
            # The recent version torch.multinomial requires num_samples arg.
            # Previously, the default was num_samples=1
            x = torch.multinomial(prob, num_samples=1).view(-1)
            # x.shape = (batch_size)
            sequences.append(x.view(-1, 1))

            one_hot_labels = nn.functional.one_hot(x, self.voc.vocab_size)
            # one_hot_labels.shape == (batch_size, vocab_size)
            likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1) ###

            x = x.data.clone()
            # is EOS sampled at a certain example?
            EOS_sampled = (x == self.eos_idx).data
            # torch.ge : greater or equal operator
            finished = torch.ge(finished + EOS_sampled.cpu(), 1)
            
            # if all the examples have produced EOS once, we will break the loop
            if torch.prod(finished) == 1: break 

        # Each element in sequences is in shape (batch_size x 1)
        # concat on dim=1 to get (batch_size x seq_len)
        sequences = torch.cat(sequences, 1)
        return sequences.data, likelihoods

    def likelihood_map(self, seq):
        """
            For a given sequence, return the probability map.
            The returned prob_map attaches the gradients, so if you are not intended to use gradients,
            it would be good idea to get the item values before use it.

            Args:
                seq: (seq_length) A single sequence in integer.
                <EOS> should be already in. <BEG> is not in.

            Outputs:
                prob_map: (vocab_size, seq_length+1) likelihood map for the entire sequence.
                with <BEG> included, and <EOS> excluded.
        """
        enc_size = len(seq)
        if seq[-1] != self.eos_idx:
            print("[Error] the input sequence should have <EOS> token at the end!!")
            return None
        enc_smiles = torch.zeros(enc_size)
        enc_smiles[0] = self.beg_idx  # add <BEG> at the beginning
        enc_smiles[1:] = seq[:-1] # truncate the <EOS> at the end
    
        hl_units = self.lstm.hidden_layer_units
        hidden_states = []
        cell_states = []
        for i in range(len(hl_units)):
            hidden_state = torch.zeros(1, hl_units[i]).float() 
            hidden_states.append(hidden_state)
            cell_state = torch.zeros(1, hl_units[i]).float()
            cell_states.append(cell_state)
        
        prob_map = torch.zeros((self.voc.vocab_size, len(enc_smiles)))
        for step in range(len(enc_smiles)):
            x = enc_smiles[step].reshape(-1).long()
            prob, _, hidden_states, cell_states = self.step_likelihood(x, hidden_states, cell_states)
            prob_map[:, step] = prob
        return prob_map

def perform_train_epoch(smtk: SmilesTokenizer, train_data_loader: DataLoader, rnn_generator: SmilesLSTMGenerator, 
                        optimizer, verbose=False):
    vocab = rnn_generator.voc
    
    loss_collection = 0.0
    for step, batch in enumerate(train_data_loader):
        if verbose and step % 1000 == 0:
            print(step)
        
        # for now, it is assumed that no smiles will raise KeyError in batch.
        seqs, _ = prepare_batch(batch, smtk, vocab)
        seqs = seqs.long() # converting float to long int

        NLLosses, _ = rnn_generator.likelihood(seqs)
        # NLLosses: (batch_size,) - each example loss
        mean_loss = NLLosses.mean() 

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()
        
        loss_collection += NLLosses.sum().cpu().detach()
        
    epoch_loss = loss_collection / len(train_data_loader.dataset)
    return epoch_loss

def truncate_EOS(batch_token_array, vocab_obj):
    """
        The RNN model sampling produces (batch_size x seq_len) of sequences.
        This function cuts off the tokens after the first <EOS> in each sample.

        Input: batch of token lists np.array(batch_size x seq_len)
        Output: truncated sequence list
    """
    bs, _ = batch_token_array.shape
    seq_list = []
    for i in range(bs):
        tokens = batch_token_array[i].tolist()
        # append EOS at the end
        tokens.append(vocab_obj.get_EOS_idx())
        # find EOS position of first encounter
        EOS_pos = tokens.index(vocab_obj.get_EOS_idx())
        # get the seq until right before EOS
        seq_list.append(tokens[0:EOS_pos])
    return seq_list

def decode_seq_list(batch_token_list, vocab_obj):
    """
        Input: batch of token(index) lists (batch_size,)
        Output: (decoded) smiles list
    """
    bs = len(batch_token_list)
    smiles_list = []
    for i in range(bs):
        tokens = batch_token_list[i]
        smiles = vocab_obj.decode(tokens)
        smiles_list.append(smiles)
    return smiles_list

def get_NLLs_batch(smiles_list, lstm_generator, smtk, vocab_obj, batch_size=128):
    """ batch process NLL calculator """
    indices = list(range(len(smiles_list)))
    idloader = DataLoader(indices, batch_size=batch_size, shuffle=False)
    nlls_list = [] # element is a batch of seq nlls
    ke_id_list = [] # indices that raised KeyErrors.
    for batch_ids in idloader:
        smis = np.array(smiles_list)[batch_ids.tolist()]
        # esmi = encoded smi
        batch_esmis, keyerror_ids = prepare_batch(smis, smtk, vocab_obj)
        # smiles raising KeyError won't be included in batch_esmis.
        ke_id_list.extend([batch_ids[index] for index in keyerror_ids])
        nlls, _ = lstm_generator.likelihood(batch_esmis)
        nlls_list.append(nlls.detach().cpu().numpy())
    nlls_cat = np.array([])
    for nll in nlls_list:
        nlls_cat = np.concatenate((nlls_cat, nll))
    return nlls_cat, ke_id_list
