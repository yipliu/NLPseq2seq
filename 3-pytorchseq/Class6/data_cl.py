import torch

#from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset

import spacy
import numpy as np

import random
import math
"""
This file is made up of the data_reader.py and is just converted to Class
"""
class Lang():
    def __init__(self, src_model: str, trg_model: str):
        """
        src_model: 'de_core_news_sm'
        trg_model: 'en_core_web_sm'
        They must be Spacy language model
        """
        self.spacy_src = spacy.load(src_model)
        self.spacy_trg = spacy.load(trg_model)

        self.SRC = Field(tokenize = self.tokenize_src, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True, 
                    include_lengths = True)

        self.TRG = Field(tokenize = self.tokenize_trg, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)

    def tokenize_src(self, text):
        return [tok.text for tok in self.spacy_src.tokenizer(text)]

    def tokenize_trg(self, text):
        return [tok.text for tok in self.spacy_trg.tokenizer(text)]

    def load_data(self):
        # train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
        #                                             fields = (self.SRC, self.TRG))

        fields = {'Src': ('s', self.SRC), 'Trg':('t',self.TRG)}
        
        train_data, test_data = TabularDataset.splits( path='data',
                                                       train='train.json',
                                                       test='test.json',
                                                       format='json',
                                                       fields = fields
                                                      )
        return train_data, test_data

    def fields_vocab(self, dataset):
        self.SRC.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)
        self.TRG.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)
