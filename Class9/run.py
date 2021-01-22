import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator, Iterator

from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

import os 
import argparse
import time
import math
import random

from utils.data_cl import Lang
from utils.model_func import *
from model import Encoder, Decoder, Attention, Seq2Seq

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='data',
                    help= 'the folder of data')
parser.add_argument('--input_train', type=str, default='train.json',
                    help='train data')
parser.add_argument('--input_test', type=str, default='test.json',
                    help='test data')
parser.add_argument('--lang_model', type=str, default='en_core_web_sm',
                    help='the model of spacy')


parser.add_argument('--enc_hidden_dim', type=int, default=256, help='')
parser.add_argument('--enc_emb_dim', type=int, default=100, help='')
parser.add_argument('--dec_hidden_dim', type=int, default=256, help='')
parser.add_argument('--dec_emb_dim', type=int, default=100, help='')
parser.add_argument('--enc_dropout', type=int, default=0.5, help='')
parser.add_argument('--dec_dropout', type=int, default=0.5, help='')
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--n_epoch', type=int, default=5, help='')

parser.add_argument('--seed', type=int, default=1234, help='')
parser.add_argument('--k_fold', type=bool, default=True, help='')
parser.add_argument('--k', type=int, default=5, help='')

opt = parser.parse_args()


def main():

    # SEED
    SEED = opt.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Model
    ENC_EMB_DIM = opt.enc_emb_dim
    DEC_EMB_DIM = opt.dec_emb_dim
    ENC_HID_DIM = opt.enc_hidden_dim
    DEC_HID_DIM = opt.dec_hidden_dim
    ENC_DROPOUT = opt.enc_dropout
    DEC_DROPOUT = opt.dec_dropout
    
    BATCH_SIZE = opt.batch_size
    N_EPOCHS = opt.n_epoch
    CLIP = 1

    N_FOLDS = opt.k
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    lang = Lang(opt.lang_model, opt.lang_model,
                opt.path, opt.input_train, opt.input_test)
    #lang = Lang(opt.lang_model, opt.lang_model, 'data', 'train.json', 'test.json')


    # Get dataset
    train_data, test_data = lang.load_data()
    print(vars(train_data[0]))
    # Build vocab
    lang.fields_vocab(train_data)

    INPUT_DIM, OUTPUT_DIM = len(lang.SRC.vocab), len(lang.TRG.vocab)
    print("The length of src_vocab is {}; \n The length of trg_vocab is {};". \
                                                            format(INPUT_DIM, OUTPUT_DIM))
    
    TRG_PAD_IDX = lang.TRG.vocab.stoi[lang.TRG.pad_token]
    SRC_PAD_IDX = lang.SRC.vocab.stoi[lang.SRC.pad_token]

    torch.save(train_data.fields, 'fields.pkl')
    print('The Fields: fields.pkl is stored in local')

    # Model
    print("====Construct the model====")
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, DEVICE).to(DEVICE)
    
    print("-------Model Parameters----")
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # ignore the loss on <pad> tokens
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    optimizer = optim.Adam(model.parameters())
    
    
    # initialize the model to a special initialization
    model.apply(init_weights)
    print("Encoder Model is {}; \n Decoder Model is {}; \n Seq2Seq model is {}". \
                                                                    format(enc, dec, model))


    # cross validation, default = True
    if opt.k_fold:
        train_val_generator, test_dataset = lang.get_cvdatasets(N_FOLDS, SEED)

        for fold, (train_dataset, val_dataset) in enumerate(train_val_generator):
            
            best_valid_loss = float('inf')
            print("FOLD:", fold + 1)
        
            train_iter, val_iter, test_iter = BucketIterator.splits(
                                                        (train_dataset, val_dataset, test_dataset), 
                                                        sort_within_batch = True,
                                                        sort_key = lambda x: len(x.Src),
                                                        batch_size = BATCH_SIZE,
                                                        shuffle = True, 
                                                        device = 'cuda')

                
            for epoch in range(N_EPOCHS):
                best_model, best_epoch,best_valid_loss = epoch_train(model, train_iter,val_iter,optimizer,
                                                                        criterion,CLIP, epoch, N_EPOCHS,best_valid_loss)
            torch.save(best_model, 'fold_' + str(fold+1) + '_epoch_' + str(best_epoch+1) + 'seqmodel')
            loop_test = tqdm(enumerate(test_iter), total=len(test_iter))
            test_loss = model_evaluate(best_model, loop_test, criterion, epoch, N_EPOCHS)
            print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')
 
    else: # normal train, test
        train_iter, test_iter = BucketIterator.splits(
                                                    (train_data, test_data),
                                                    sort_within_batch=True,
                                                    sort_key= lambda x: len(x.s),
                                                    batch_size = BATCH_SIZE,
                                                    shuffle = True, 
                                                    device= 'cuda'
                                                    )

        for epoch in range(N_EPOCHS):
            best_model, best_epoch = epoch_train(model, train_iter,val_iter,optimizer,criterion,CLIP, epoch, N_EPOCHS)
        torch.save(best_model, 'epoch_' + str(best_epoch+1) + 'seqmodel')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print('[STOP]', e)