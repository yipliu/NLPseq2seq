import torch
import torch.nn as nn

import torch.optim as optim

from data_reader import load_datasets
from utils import train, evaluate
from model import Encoder, Decoder, Seq2Seq

from tqdm import tqdm

import time
import math

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SIZE = 128

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    assert torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("====> prepare Datasets")
    train_iter, val_iter, test_iter, DE, EN = load_datasets(BATCH_SIZE, device)
    #len_train, len_val, len_test = len(train_iter), len(val_iter), len(test_iter)
    print("====> Datasets has been prepared")

    INPUT_DIM, OUTPUT_DIM = len(DE.vocab), len(EN.vocab) 
    print("INPUT_DIM is {}".format(INPUT_DIM))
    print("OUTPUT_DIM is {}".format(OUTPUT_DIM))

    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = EN.vocab.stoi[EN.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    print("Begining to train the model...")
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        loop_train = tqdm(enumerate(train_iter), total=len(train_iter))
        
        # train_loss: the loss that is averaged over all batches
        train_loss = train(model, loop_train, optimizer, criterion, CLIP, epoch, N_EPOCHS)
        
        loop_valid = tqdm(enumerate(val_iter), total=len(val_iter))
        valid_loss = evaluate(model, loop_valid, criterion, epoch, N_EPOCHS)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tutl-model.pt')
    
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss: .3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss: .3f} | Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)