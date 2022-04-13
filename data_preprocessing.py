import numpy as np
import pandas as pd
import string

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

class RaceDataset(Dataset):

    def __init__(self, split: string):
        # Load and convert dataset from pyarrow to pandas df
        self.dataset = load_dataset('race', 'all')[split].to_pandas()
        self.processed = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset.iloc[i]

    def drop_numeric(self, df):
        # Map letter answers to options list indices
        df['answer'] = df['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
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
        val_df = self.drop_numeric(self.dataset)
        # Rule 2
        # val_df = 
        # Etc...
        self.processed = True
        return val_df

