import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math



def load_data(batch_size, device):

    spacy_de, spacy_en = spacy.load('de_core_news_sm'), spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

        
    # include_lengths: this will make batch.src to be a tuple.
    # (a batch of numericalized source sentence, the non-padded lengths)

    SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            include_lengths = True)
    

    TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields = (SRC, TRG))


    # building vocab
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
                                        (train_data, valid_data, test_data),
                                        sort_within_batch= True,
                                        sort_key=lambda x: len(x.src),
                                        batch_size = batch_size,
                                        device = device)

    #return train_iterator, valid_iterator, test_iterator, TRG_PAD_IDX, SRC_PAD_IDX, (len(SRC.vocab), len(TRG.vocab))

    #torch.save(train_data.fields, 'fields.pkl')
    return train_iterator, valid_iterator, test_iterator, (TRG_PAD_IDX, SRC_PAD_IDX), (SRC, TRG)
"""
All elements in the batch need to be sorted by their non-padded lengths in descending order

sort_within_batch: Tell the iterator that the content of the batch need to be sorted

sort_key: tell the iterator how to sort the elements in the batch.
Here, we sort by the length of the src sentence
"""

    