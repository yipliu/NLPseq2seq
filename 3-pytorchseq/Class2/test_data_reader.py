import torch
from data_reader import load_data


if __name__ == "__main__":
    BATCH_SIZE = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator, len1, len2 = load_data(BATCH_SIZE, device)
    print(len1)
    print(len2)