import numpy as np

# If you use Vocabulary.add_tokens(), the dictionary will completely be changed.
# That means you will have different index value for an already-existing token.
# init_from_file() will remove the current vocab and reinitialize all the things.
class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, max_length=150, init_from_file=None):
        self.special_tokens = ['<PAD>', '<BEG>', '<EOS>'] 
        # we will add these special tokens if they are not already in the tokens file
        self.token_set = set()
        self.token_set = self.token_set.union(set(self.special_tokens))
        self.vocab_size = len(self.token_set)
        self.vocab = dict(zip(self.token_set, range(self.vocab_size)))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        self.max_length = max_length
        if init_from_file:
            self.init_from_file(init_from_file)

    def get_BEG_idx(self): return self.vocab['<BEG>']
    def get_EOS_idx(self): return self.vocab['<EOS>']
    def get_PAD_idx(self): return self.vocab['<PAD>']

    def add_tokens(self, tokens):
        """Adds characters to the vocabulary"""
        self.token_set = self.token_set.union(tokens)
        self.vocab_size = len(self.token_set)
        self.vocab = dict(zip(self.token_set, range(self.vocab_size)))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        # Try to follow the token orders in the file...
        with open(file, 'r') as f:
            tokens = f.read().split()
        self.vocab = dict(zip(tokens, range(len(tokens))))
        temp_vs = len(self.vocab)
        for spt in self.special_tokens:
            if spt not in self.vocab.keys():
                self.vocab[spt] = temp_vs
                temp_vs += 1
        self.token_set = set(self.vocab.keys())
        self.vocab_size = len(self.token_set)
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def have_invalid_token(self, token_list):
        for i, token in enumerate(token_list):
            if token not in self.vocab.keys():
                return True
        return False

    def encode(self, token_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        if type(token_list) != list:
            print("encode(): the input was not a list type!!!")
            return None
        token_indices = np.zeros(len(token_list), dtype=np.int32)
        for i, token in enumerate(token_list):
            try:
                token_indices[i] = self.vocab[token]
            except KeyError as err:
                # print("encode(): KeyError occurred! %s"%err)
                raise
        return token_indices

    def decode(self, token_indices):
        """Takes an array of indices and returns the corresponding SMILES"""
        tokens = []
        for i in token_indices:
            tokens.append(self.reversed_vocab[i])
        smiles = "".join(tokens)
        return smiles

class SmilesTokenizer(object):
    def __init__(self, vocab_obj: Vocabulary):
        self.vocab_obj = vocab_obj
        self.multi_chars = set()
        for token in vocab_obj.token_set:
            if len(token) >= 2 and token not in vocab_obj.special_tokens:
                self.multi_chars.add(token)

    def tokenize(self, smiles):
        """Return a list of tokens"""
        # start with spliting with multi-char tokens
        token_list = [smiles] 
        for k_token in self.multi_chars:
            new_tl = []
            for elem in token_list:
                sub_list = []
                # split the sub smiles with the multi-char token
                splits = elem.split(k_token)
                # sub_list will have multi-char token between each split
                for i in range(len(splits) - 1):
                    sub_list.append(splits[i])
                    sub_list.append(k_token)
                sub_list.append(splits[-1]) 
                new_tl.extend(sub_list)
            token_list = new_tl
    
        # Now, only one-char tokens to be parsed remain.
        new_tl = []
        for token in token_list:
            if token not in self.multi_chars:
                new_tl.extend(list(token))
            else:
                new_tl.append(token)
        # Note that invalid smiles characters can be produced, if the smiles contains un-registered characters.
        return new_tl

def locate_specials(vocab: Vocabulary, seq):
    """Return special token (BEG, EOS, PAD) positions in the token sequence"""
    spinds = [vocab.get_BEG_idx(), vocab.get_EOS_idx(), vocab.get_PAD_idx()]
    special_pos = []
    for i, token in enumerate(seq):
        if token in spinds:
            special_pos.append(i)
    return special_pos
