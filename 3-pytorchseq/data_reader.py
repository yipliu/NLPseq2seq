import torch



from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math


"""
1. Aim: Set the random seeds for deterministic results
"""
# SEED = 1234 

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

def load_datasets(batch_size, device):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
   
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
  
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # Creating Field
    DE = SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True )
    EN = TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True )

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                        fields = (SRC, TRG))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    # vars: 
    print(vars(train_data.examples[0]))

    print("type of train_data is {}".format(type(train_data)))
    print("type of train_data.examples[0] is {}".format(type(train_data.examples[0])))

    # Building Vocabulary
    SRC.build_vocab(train_data, min_freq = 2 )
    TRG.build_vocab(train_data, min_freq = 2 )

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size = batch_size, device = device)

    return train_iterator, valid_iterator, test_iterator, DE, EN