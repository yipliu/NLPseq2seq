"""
This code is from Ceshine Lee, for detailed information you can find from url:
https://towardsdatascience.com/use-torchtext-to-load-nlp-dataset-part-ii-f146c8b9a496
"""
import torch
import spacy
import numpy as np
import pandas as pd
from torchtext.data import Field, TabularDataset, Dataset, BucketIterator
from sklearn.model_selection import KFold

from joblib import Memory
"""
Aim: Realize the cross-validtion methods based on torchtext

- TabularDataset: Defines a Dataset of columns stored in CSV, TSV, JSON format

- Dataset: Defines a dataset composed of Examples along with its Fields

Plan:

Dataset is JSON format

1, Using TabularDataset to build a dataset
2. Using train.examples to get Examples, it is a list.
3. Using KFold to split the Examples
4. Using Dataset to define a datset based on the sub-Examples.

Note: this file does not implement speed up
"""
class Lang():
    def __init__(self, src_model: str, trg_model: str, dataset_path , train_name, test_name):
        
        self.spacy_src = spacy.load(src_model)
        self.spacy_trg = spacy.load(trg_model)
        self.dataset_path = dataset_path
        self.train_name = train_name
        self.test_name = test_name

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
        """
        return Example (list)
        """

        fields = {'Src': ('Src',self.SRC), 'Trg':('Trg',self.TRG)}



        train_data, test_data = TabularDataset.splits( path=self.dataset_path,
                                                train=self.train_name,
                                                test=self.test_name,
                                                format='json',
                                                fields = fields
                                                )
        #SRC.build_vocab(train_data, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)
        #TRG.build_vocab(train_data, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)

        return train_data, test_data

    def fields_vocab(self, dataset):
        self.SRC.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)
        self.TRG.build_vocab(dataset, vectors = "glove.6B.100d", min_freq=2, unk_init= torch.Tensor.normal_)


    def get_datasets(self, n_folds, SEED):
        
        train_data, test_data = self.load_data()

        train_exs, test_exs = train_data.examples, test_data.examples

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

        fields = [('Src', self.SRC), ('Trg', self.TRG)]

        def iter_folds():
            train_exs_arr = np.array(train_exs)
            for train_idx, val_idx in kf.split(train_exs_arr):
                yield(
                    Dataset(train_exs_arr[train_idx], fields),
                    Dataset(train_exs_arr[val_idx], fields),
                )
        
        test_d = Dataset(test_exs, fields)
        return iter_folds(), test_d

# if __name__ == "__main__":
#     lang = Lang_cf('en_core_web_sm','en_core_web_sm','data')
   
#     train_exs, test_exs, SRC, TRG = lang.read_files()
#     print(SRC.vocab.stoi['the'])
#     print(SRC.vocab.itos[10])