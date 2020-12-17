Reference:
1. [keon-seq2seq](https://github.com/keon/seq2seq)
2. [bentrevett](https://github.com/bentrevett/pytorch-seq2seq)

Tips from bentrevett:
 ***tqdm***

for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
- leave=False -> one line
- leave=True  -> many lines