import numpy as np
import pandas as pd
import string

import torch
from torch.utils.data import Dataset
from util_methods import encode_data

from datasets import load_dataset

class RaceDataset(Dataset):

    def __init__(self, split: string):
        # Load and convert dataset from pyarrow to pandas df
        self.dataset = load_dataset('race', 'all')[split].to_pandas().iloc[:1000]
        self.processed = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {
                  'input_ids':(self.dataset['input_ids'][i]), 
                  'attention_mask':(self.dataset['attention_mask'][i]),
                  'labels':self.dataset['label'][i]
                }

    def drop_numeric(self, df):
        # Map letter answers to options list indices
        df['answer'] = df['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
        # Replace list comprehension for speed
        ans_list = []
        print(df['answer'].iat[1])

        for i in range(len(df['answer'])):
            # print(df['answer'].iat[i])
            # print(df['options'].iat[i][2])
            df_ans_i = df['answer'].iat[i]
            ans_list.append((df['options'].iat[i])[df_ans_i])

        df['answer'] = pd.Series(ans_list)
        print(df['answer'].iat[1])
        options_series = df['options']

        def is_valid_options(options_list):
            # Strip option, remove punctuation, return True if not all options are just numbers.
            is_valid = [not sent.strip().translate(str.maketrans('', '', string.punctuation)).isnumeric() for sent in options_list]
            return any(is_valid)

        # Map is_valid_options to every question
        is_valid_series = options_series.map(is_valid_options)
        num_dropped = is_valid_series.value_counts()[False]
        print(f'Dropped {num_dropped} rows with only numeric answers')

        return df[is_valid_series]

    def preprocess_dataset(self):
        # Continue to add preprocessing steps
        # Rule 1
        df = self.drop_numeric(self.dataset)
        # Rule 2
        # df = 
        # Etc...
        self.processed = True
        self.dataset = df

