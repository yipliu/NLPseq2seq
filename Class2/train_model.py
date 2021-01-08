import torch
import torch.nn as nn
import torch.optim as optim

from data_reader import load_data
from model import Encoder, Decoder, Seq2Seq
from utils import epoch_time, model_train, model_evaluate

from tqdm import tqdm

import time
import math

# initialize the model
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

# counting the parameter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    BATCH_SIZE = 128
    N_EPOCHS = 10
    CLIP = 1

    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_valid_loss = float('inf')


    print("===== Prepare the dataset====")
    train_iter, val_iter, test_iter, TRG_PAD_IDX, len_vocab= load_data(BATCH_SIZE, device)
    print("The length of src_vocab is {}; \n The length of trg_vocab is {};". \
                                                            format(len_vocab[0], len_vocab[1]))

    INPUT_DIM, OUTPUT_DIM = len_vocab[0], len_vocab[1]
   
    print("====Construct the model====")
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # initialize the model to a special initialization
    model.apply(init_weights)
    print("Encoder Model is {}; \n Decoder Model is {}; \n Seq2Seq model is {}". \
                                                                    format(enc, dec, model))
                    

    print("-------Model Parameters----")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    # ignore the loss on <pad> tokens
    
    criterion = nn.CrossEntropyLoss(ignore_index= TRG_PAD_IDX)


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
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
    

