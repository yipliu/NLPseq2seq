import torch
from data_reader import load_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
train_iter, val_iter, test_iter, DE, EN = load_datasets(BATCH_SIZE, device)