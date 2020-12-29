import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split
"""
Translate txt into json
"""

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def processData(src_pat, trg_path, encording_type):
    src_lines = open(src_pat, encoding= encording_type).read().split('\n')
    trg_lines = open(trg_path, encoding= encording_type).read().split('\n')

    assert len(src_lines) == len(trg_lines), \
        "Length of src and trg must be equal!"

    raw_data = {'Src': [normalizeString(line) for line in src_lines],
                'Trg': [line for line in trg_lines]}

    return raw_data


if __name__ == "__main__":

    src_path, trg_path = 'data/hard_pc_src.txt', 'data/hard_pc_tar.txt'

    raw_data = processData(src_path, trg_path, 'ISO-8859-1')


    df = pd.DataFrame(raw_data, columns=['Src', 'Trg'])

    # split train and test
    # In 2019, using 20% for test, using 80% for 5 fold cross-validation  
    train_data, test_data = train_test_split(df, test_size=0.2)

    # save the data in json type
    train_data.to_json('data/train.json', orient='records', lines=True)
    test_data.to_json('data/test.json', orient='records', lines=True)
