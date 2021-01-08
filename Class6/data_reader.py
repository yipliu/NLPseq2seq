from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, TabularDataset

import spacy
import numpy as np

import random
import math



def load_data(batch_size, device):

    spacy_en = spacy.load('en_core_web_sm')
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

        
    # include_lengths: this will make batch.src to be a tuple.
    # (a batch of numericalized source sentence, the non-padded lengths)

    english = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                include_lengths = True)
    

    command = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    fields = {'Src': ('s', english), 'Trg':('t',command)}

    train_data, test_data = TabularDataset.splits(
        path='data',
        train='train.json',
        test='test.json',
        format='json',
        fields = fields
    )

    print(vars(train_data[0]))

    # building vocab
    english.build_vocab(train_data, min_freq=2)
    command.build_vocab(train_data, min_freq=2)
    
    
    train_iterator, test_iterator = BucketIterator.splits(
                                        (train_data, test_data),
                                        sort_within_batch= True,
                                        sort_key=lambda x: len(x.src),
                                        batch_size = batch_size,
                                        device = device)

    TRG_PAD_IDX = command.vocab.stoi[command.pad_token]
    SRC_PAD_IDX = english.vocab.stoi[english.pad_token]


    #return train_iterator, valid_iterator, test_iterator, TRG_PAD_IDX, SRC_PAD_IDX, (len(SRC.vocab), len(TRG.vocab))

    #torch.save(train_data.fields, 'fields.pkl')
    return train_iterator, test_iterator, (TRG_PAD_IDX, SRC_PAD_IDX), (english, command)
"""
All elements in the batch need to be sorted by their non-padded lengths in descending order

sort_within_batch: Tell the iterator that the content of the batch need to be sorted

sort_key: tell the iterator how to sort the elements in the batch.
Here, we sort by the length of the src sentence
"""

    