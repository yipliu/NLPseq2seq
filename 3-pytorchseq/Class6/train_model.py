import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator

from data_reader import load_data
from data_cl import Lang
from model import Encoder, Decoder, Seq2Seq, Attention
from utils import epoch_time, model_train, model_evaluate, translate_sentence

from tqdm import tqdm
import numpy as np

import time
import math
import random

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():

    SEED = 1234
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ENC_EMB_DIM = 100
    DEC_EMB_DIM = 100
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SIZE = 128

    N_EPOCHS = 1000
    CLIP = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("===== Prepare the dataset====")
    #train_iter, val_iter, test_iter, (TRG_PAD_IDX, SRC_PAD_IDX), (SRC, TRG)= load_data(BATCH_SIZE, device)
    
    language_model = Lang('en_core_web_sm', 'en_core_web_sm')
    train_data, test_data = language_model.load_data()
    print(vars(train_data[0]))
    #language_model.SRC.build_vocab(train_data, min_freq=2)
    #language_model.TRG.build_vocab(train_data, min_freq=2)
    # Build vocabulary
    language_model.fields_vocab(train_data)

    train_iter, test_iter = BucketIterator.splits(
                                        (train_data, test_data), 
                                        sort_within_batch = True,
                                        sort_key = lambda x: len(x.s),
                                        batch_size = BATCH_SIZE, 
                                        device = 'cuda')
    
    INPUT_DIM, OUTPUT_DIM = len(language_model.SRC.vocab), len(language_model.TRG.vocab) 

    print("The length of src_vocab is {}; \n The length of trg_vocab is {};". \
                                                            format(INPUT_DIM, OUTPUT_DIM))
    
      
    #train_data = train_iter.dataset
    #fids = torch.save(train_data.fields, 'fields.pkl')
   
    TRG_PAD_IDX = language_model.TRG.vocab.stoi[language_model.TRG.pad_token]
    SRC_PAD_IDX = language_model.SRC.vocab.stoi[language_model.SRC.pad_token]
    torch.save(train_data.fields, 'fields.pkl')
    print('The Fields: fields.pkl is stored in local')
    # create and initialize the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)




   
    print("====Construct the model====")
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

    # initialize the model to a special initialization
    model.apply(init_weights)
    print("Encoder Model is {}; \n Decoder Model is {}; \n Seq2Seq model is {}". \
                                                                    format(enc, dec, model))
    

    print("-------Model Parameters----")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    # ignore the loss on <pad> tokens
    
    criterion = nn.CrossEntropyLoss(ignore_index= TRG_PAD_IDX)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        loop_train = tqdm(enumerate(train_iter), total=len(train_iter))
        train_loss = model_train(model, loop_train, optimizer, criterion, CLIP, epoch, N_EPOCHS)
        
        loop_valid = tqdm(enumerate(test_iter), total=len(test_iter))
        valid_loss = model_evaluate(model, loop_valid, criterion, epoch, N_EPOCHS)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), 'tut2-model.pt')
            torch.save(model,'seqmodel')    
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    #### eval model
    # example_idx = 12

    # src = vars(train_iter.dataset.examples[example_idx])['src']
    # trg = vars(train_iter.dataset.examples[example_idx])['trg']
    # print('The src sentence is {}, \n the trg sentence is {}'.format(src, trg))

    # translation, attention = translate_sentence(src, 'de_core_news_sm', SRC, TRG, model, device)
    # print(f'predicted trg = {translation}')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
    