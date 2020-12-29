import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math

class Lang():
    def __init__(self, src_model, trg_model):
        self.src_model = src_model
        self.trg_model = trg_model

        self.spacy_de = spacy.load(self.src_model)
        self.spacy_en = spacy.load(self.trg_model)

        self.SRC = Field(tokenize = self.tokenize_de, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True, 
                    include_lengths = True)

        self.TRG = Field(tokenize = self.tokenize_en, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True)

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def load_data(self):
        train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (self.SRC, self.TRG))
        return train_data, valid_data, test_data

