import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator, Iterator

from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

import time 
import math
import random

from utils.data_cl import Lang
from utils.model_func import *
from model import Encoder, Decoder, Attention, Seq2Seq


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
    BATCH_SIZE = 2

    N_EPOCHS = 3
    CLIP = 1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lang = Lang('en_core_web_sm','en_core_web_sm','data', 'train.json', 'test.json')
    # Get dataset
    train_data, test_data = lang.load_data()
    print(vars(train_data[0]))
    # build vocab
    lang.fields_vocab(train_data)

    train_val_generator, test_dataset = lang.get_datasets(5, SEED)

    INPUT_DIM, OUTPUT_DIM = len(lang.SRC.vocab), len(lang.TRG.vocab)
    print("The length of src_vocab is {}; \n The length of trg_vocab is {};". \
                                                            format(INPUT_DIM, OUTPUT_DIM))

    TRG_PAD_IDX = lang.TRG.vocab.stoi[lang.TRG.pad_token]
    SRC_PAD_IDX = lang.SRC.vocab.stoi[lang.SRC.pad_token]
    
    torch.save(train_data.fields, 'fields.pkl')
    print('The Fields: fields.pkl is stored in local')

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    print("====Construct the model====")
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, DEVICE).to(DEVICE)

    print("-------Model Parameters----")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    # ignore the loss on <pad> tokens
    

    best_valid_loss = float('inf')

    # initialize the model to a special initialization
    model.apply(init_weights)
    print("Encoder Model is {}; \n Decoder Model is {}; \n Seq2Seq model is {}". \
                                                                    format(enc, dec, model))

    for fold, (train_dataset, val_dataset) in enumerate(train_val_generator):
        print("FOLD:", fold + 1)
        
        train_iter, val_iter, test_iter = BucketIterator.splits(
                                        (train_dataset, val_dataset, test_dataset), 
                                        sort_within_batch = True,
                                        sort_key = lambda x: len(x.Src),
                                        batch_size = BATCH_SIZE,
                                        shuffle = True, 
                                        device = 'cuda')
    
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            loop_train = tqdm(enumerate(train_iter), total=len(train_iter))
            train_loss = model_train(model, loop_train, optimizer, criterion, CLIP, epoch, N_EPOCHS)
            
            loop_valid = tqdm(enumerate(val_iter), total=len(val_iter))
            valid_loss = model_evaluate(model, loop_valid, criterion, epoch, N_EPOCHS)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #torch.save(model.state_dict(), 'tut2-model.pt')
                torch.save(model, str(fold) + 'seqmodel')
                model_best = model    
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
        loop_test = tqdm(enumerate(test_iter), total=len(test_iter))
        test_loss = model_evaluate(model_best, loop_test, criterion, epoch, N_EPOCHS)
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')

        


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

