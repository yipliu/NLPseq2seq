import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math


def create_field(de, en):

    spacy_de, spacy_en = spacy.load(de), spacy.load(en)


    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

    TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

    return SRC, TRG

def load_data(batch_size, device):
   
    SRC, TRG = create_field('de_core_news_sm', 'en_core_web_sm')


    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields = (SRC, TRG))


    # building vocab
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_within_batch= True,
    sort_key=lambda x: len(x.src),
    batch_size = batch_size,
    device = device)

    return train_iterator, valid_iterator, test_iterator, TRG_PAD_IDX, (len(SRC.vocab), len(TRG.vocab))