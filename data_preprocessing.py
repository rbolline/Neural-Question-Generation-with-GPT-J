import numpy as np
import pandas as pd
import string

import torch
from torch.utils.data import Dataset
from util_methods import encode_data

from transformers import RobertaTokenizerFast
from datasets import load_dataset

class RaceDataset(Dataset):

    def __init__(self, split: string, tokenizer):
        # Load and convert dataset from pyarrow to pandas df
        self.dataset = load_dataset('race', 'all')[split].to_pandas()
        self.dataset = self.dataset.sample(frac=0.12, random_state=1).reset_index(drop=True)
        # print(self.dataset.iloc[:5])
        self.encoded_data = []
        self.tokenizer = tokenizer
        self.labels = []
        self.processed = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
                  'input_ids':self.encoded_data['input_ids'][i], 
                  'attention_mask':self.encoded_data['attention_mask'][i],
                  'labels':self.labels[i]
                }

    def drop_numeric(self, df):
        # Map letter answers to options list indices
        df['answer'] = df['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
        # Replace list comprehension for speed
        ans_list = []
        
        for i in range(len(df['answer'])):
            # print(df['answer'].iat[i])
            # print(df['options'].iat[i][2])
            df_ans_i = df['answer'].iat[i]
            ans_list.append((df['options'].iat[i])[df_ans_i])

        df['answer'] = pd.Series(ans_list)
        # print(df['answer'].iat[1])
        options_series = df['options']

        def is_valid_options(options_list):
            # Strip option, remove punctuation, return True if not all options are just numbers.
            is_valid = [not sent.strip().translate(str.maketrans('', '', string.punctuation)).isnumeric() for sent in options_list]
            return any(is_valid)

        # Map is_valid_options to every question
        is_valid_series = options_series.map(is_valid_options)
        #num_dropped = is_valid_series.value_counts()[False]

        return df[is_valid_series]

    def preprocess_dataset(self):
        # Continue to add preprocessing steps  
        # Rule 1: Drop numeric
        val_df = self.drop_numeric(self.dataset)
        
        ## Rule 2: Exclude phrase completion questions
        val_df = val_df[ ~val_df['question'].str.contains('_')]

        ## Rule 3: Exclude "According to the passage" questions
        val_df = val_df[ ~val_df['question'].str.contains('According to the passage')]

        ## Rule 4: Exclude questions shorter than 5 words
        val_df = val_df[ val_df.question.str.replace(',','').str.split().str.len() > 5 ]
        
        ## Rule 5: Added filtering for contexts with less than n associated questions
        n = 1
        vc = filt_df['example_id'].value_counts().to_frame()
        vc.columns = ['Count']
        vc[vc.Count > n]
        filt_df = filt_df[filt_df['example_id'].isin(vc.index)]
        
        self.dataset = val_df
   


        # print((self.dataset['answer'].iloc[:5]))
        self.encoded_data, self.labels = encode_data(self.dataset, self.tokenizer)
        self.labels = self.labels.tolist()
        self.processed = True
    def test_dataset(self, dataset):
        self.encoded_data, self.labels = encode_data(dataset, self.tokenizer)
        self.labels = self.labels.tolist()

