import torch

from utils import translate_sentence

#model = torch.load('tut2-model.pt')
model = torch.load('seqmodel')
fields = torch.load('fields.pkl')
print(fields)
src = fields['src']
trg = fields['trg']
with open('source.txt') as f:
    source = f.read()
print(source)

tar, attention = translate_sentence(source, 'de_core_news_sm', src, trg, model,'cuda')
print(tar)


    
    
