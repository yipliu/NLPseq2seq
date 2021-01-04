# PyTorch Seq2Seq
This repo implements seq2seq models using PyTorch and TorchText

# Novel functions in this repo
- tqdm  
- 5-fold cross validation
- Save/Load Model, Field

# TODO
- Plot result
- Beautify the code

# Important
1. For TabularDataset: using fields = {'Src': ('Src',SRC), 'Trg':('Trg',TRG)}

2. For Dataset: using fields = [('Src', SRC), ('Trg', TRG)]

**Make Sure** they have the same fiedls and according with JSOM files


**Reference**
1. 5-fold cross validation
- [Theory-1](https://scikit-learn.org/stable/modules/cross_validation.html?highlight=cross_validation)
- [Theory-2](https://machinelearningmastery.com/k-fold-cross-validation/)
- [Code]((https://towardsdatascience.com/use-torchtext-to-load-nlp-dataset-part-ii-f146c8b9a496))

2. Thanks [bentrevett](https://github.com/bentrevett/pytorch-seq2seq) for seq2seq model

3. Thanks [Aladdinpersson](https://github.com/aladdinpersson) for tqdm and Youtube video

4. Thaks Thao 