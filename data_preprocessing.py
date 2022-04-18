import numpy as np
import pandas as pd
import string

import torch
from torch.utils.data import Dataset
from util_methods import encode_data

from datasets import load_dataset

class RaceDataset(Dataset):

    def __init__(self):
        # Load and convert dataset from pyarrow to pandas df
        self.dataset = load_dataset('race', 'all')
        self.processed = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset.iloc[i]

    def get_split(self, split, do_preprocess=True):
        """Fetches train/validation/test split

        Args:
            split: string. Name of split to index

        """
        df_split = self.dataset[split].to_pandas()

        if do_preprocess:
            preprocessed_df_split = self.preprocess_dataset(df_split)
            # filtered_df_split = self.filter_dataset(preprocessed_df_split)
            # return filtered_df_split
            return preprocessed_df_split

        else:
            return df_split

    def preprocess_dataset(self, df_split):
        """Performs basic preprocessing operations on df_split

        Args:
            df_split: pd.DataFrame
        """
        df_split['answer'] = df_split['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
        df_split['answer'] = df_split.apply(lambda row: row['options'][row['answer']], axis=1)
        df_split['answer'] = df_split['answer'].str.strip('.')

        df_split['article'] = df_split['article'].str.strip('.')

        df_split['difficulty_label'] = np.where(df_split['example_id'].str.contains("high"), "Hard", "Easy")

        return df_split

    def filter_dataset(self, df_split):
        # Continue to add preprocessing steps
        
        ## Rule 1: Drop numeric
        df = self.drop_numeric(df_split)
        
        ## Rule 2: Exclude phrase completion questions
        df = df[ ~df['question'].str.contains('_')]

        ## Rule 3: Exclude "According to the passage" questions
        df = df[ ~df['question'].str.contains('According to the passage')]

        ## Rule 4: Exclude questions shorter than 5 words
        df = df[ df.question.str.replace(',','').str.split().str.len() > 5 ]
        
        self.processed = True
        self.dataset = df

    def drop_numeric(self, df):
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

    def prepare_data_qg(self, df_split, num_examples, is_train):
        '''Prepares the dataset for the Question Generation Task'''
        # prompt format <CONTEXT>\nDifficulty: <label>. Answer: <ANS>. Question: <QUES>

        def create_prompt(group):
            ##TODO: if num examples is less than len(group) then return multiple instances of prompts with context included.
            # Current code implements just 1
            if not is_train and num_examples > 1:
                raise ValueError(f"num_examples cannot be > 1 in test mode. Input: {num_examples}")

            if is_train:
                group['prompt'] = group.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}. Question: {row['question']}", axis=1)
            else:
                group['prompt'] = group.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}.", axis=1)

            context = group['article'].iloc[0]
            if num_examples >= len(group):
                with_replacement = True

            else:
                with_replacement = False

            num_reps = max(1, len(group) - num_examples)

            prompt_collection = []
            for ii in range(num_reps):
                prompt = "\n".join(group.sample(num_examples, replace=with_replacement)['prompt'].tolist())
                prompt = f"{context}\n{prompt}"

                prompt_collection.append(prompt)

            return prompt_collection

        prompt_collection = []
        for _, group in df_split.groupby('example_id'):
            prompt_collection += create_prompt(group)

        return prompt_collection
