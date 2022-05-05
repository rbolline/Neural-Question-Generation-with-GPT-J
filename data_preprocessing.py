import numpy as np
import pandas as pd
import string

import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
import argparse

from datasets import load_dataset

class RaceDataset(Dataset):

    def __init__(self):
        # Load and convert dataset from pyarrow to pandas df
        self.dataset = load_dataset('race', 'all')
        self.processed = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset.iloc[[i]]

    def get_split(self, split, do_preprocess=True):
        """Fetches train/validation/test split

        Args:
            split: string. Name of split to index

        """
        df_split = self.dataset[split].to_pandas()

        if do_preprocess:
            preprocessed_df_split = self.preprocess_dataset(df_split)
            filtered_df_split = self.filter_dataset(preprocessed_df_split)

            return filtered_df_split

        else:
            return df_split

    def preprocess_dataset(self, df_split):
        """Performs basic preprocessing operations on df_split

        Args:
            df_split: pd.DataFrame
        """
        df_split.loc[:, 'answer'] = df_split['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
        df_split.loc[:, 'answer'] = df_split.apply(lambda row: row['options'][row['answer']], axis=1)
        df_split.loc[:, 'answer'] = df_split['answer'].str.strip('.')

        df_split.loc[:, 'article'] = df_split['article'].str.strip('.')

        df_split.loc[:, 'difficulty_label'] = np.where(df_split['example_id'].str.contains("high"), "Hard", "Easy")

        # store the id of each context passage
        df_split.loc[:, 'context_id'] = df_split['example_id']

        # create unique ids for each context question pair
        df_split.loc[:, 'rolling_count'] = df_split.groupby((df_split['example_id'] != df_split['example_id'].shift(1)).astype(int).cumsum()).cumcount()
        df_split.loc[:, 'example_id'] = df_split['example_id'].str.split(".").str[0] + "_" + df_split['rolling_count'].astype(str)

        return df_split

    def filter_dataset(self, df_split):
        # Continue to add preprocessing steps

        ## Rule 1: Drop numeric
        # df_split = self.drop_numeric(df_split)

        ## Rule 2: Exclude phrase completion questions
        # df_split = df_split[ ~df_split['question'].str.contains('_')]

        ## Rule 3: Exclude "According to the passage" questions
        # df_split = df_split[ ~df_split['question'].str.contains('According to the passage')]

        ## Rule 4: Exclude answers shorter than 5 words
        df_split = df_split.loc[ df_split['answer'].str.split().str.len() > 3 ]

        ## TODO: add flag for zero-shot
        ## Drop passages that have just less than 3 questions left after filtering
        context_count_df = df_split.groupby("context_id").size().to_frame('count').reset_index()
        context_ids_to_drop = context_count_df.loc[context_count_df['count'] < 4, 'context_id'].tolist()

        df_split = df_split.loc[ ~df_split['context_id'].isin(context_ids_to_drop)]

        self.processed = True
        return df_split

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

    def prepare_data_qg(self, df_split, num_examples, is_train, is_answer_focused):
        '''Prepares the dataset for the Question Generation Task'''
        # prompt format <CONTEXT>\nDifficulty: <label>. Answer: <ANS>. Question: <QUES>

        def create_prompt(group, num_examples, is_train, is_answer_focused):
            """Cretes prompts for each article"""
            example_ids = group['example_id'].tolist()
            prompt_collection = []
            for example_id in example_ids:
                example_cond = group['example_id'].isin([example_id])
                include_df = group.loc[~example_cond]
                example_df = group.loc[example_cond]

                if is_train:
                    # if in training mode then include the question for all examples

                    if is_answer_focused:
                        include_df.loc[:, 'prompt'] = include_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}. Question: {row['question']}", axis=1)
                        example_df.loc[:, 'prompt'] = example_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}. Question: {row['question']}", axis=1)
                    else:
                        include_df.loc[:, 'prompt'] = include_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Question: {row['question']}", axis=1)
                        example_df.loc[:, 'prompt'] = example_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Question: {row['question']}", axis=1)

                else:
                    # if in test mode then include the questions for all examples except the test example.
                    if is_answer_focused:
                        include_df.loc[:, 'prompt'] = include_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}. Question: {row['question']}", axis=1)
                        example_df.loc[:, 'prompt'] = example_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Answer: {row['answer']}. Question:", axis=1)
                    else:
                        include_df.loc[:, 'prompt'] = include_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Question: {row['question']}", axis=1)
                        example_df.loc[:, 'prompt'] = example_df.apply(lambda row: f"Difficulty: {row['difficulty_label']}. Question:", axis=1)


                # include the questions from all other examples for the given context
                # first include questions that are not repeated
                num_unique_questions = min(len(include_df), num_examples)
                prompt_examples = include_df.sample(num_unique_questions, replace=False)['prompt'].tolist()

                try:
                    if num_unique_questions < num_examples:
                        diff = num_examples - num_unique_questions
                        num_rep_questions = include_df.sample(diff, replace=False)['prompt'].tolist()
                        prompt_examples += num_rep_questions
                except ValueError as err:
                    raise err
                    # print(f"num_unique_questions={num_unique_questions} num_examples={num_examples} {num_unique_questions < num_examples}, diff={diff}")

                prompt = "\n".join(prompt_examples)

                # include the question for the given example
                prompt = f"{prompt}\n{example_df['prompt'].iloc[0]}"

                # add the context article to the prompt
                context = group['article'].iloc[0]
                prompt = f"{context}\n{prompt}"

                example_df.loc[:, 'prompt'] = prompt
                prompt_collection.append(example_df)

            return prompt_collection

        prompt_collection = []
        for _, group in df_split.groupby('context_id'):
            prompt_collection += create_prompt(group,
                                               num_examples,
                                               is_train,
                                               is_answer_focused)


        prompt_df = pd.concat(prompt_collection, axis=0)
        prompt_df = prompt_df.sort_values('prompt')
        self.dataset = prompt_df

    def collate_fn(self, batch):
        batch_df = pd.concat(batch, axis=0)
        return batch_df.to_dict('list')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")
    parser.add_argument('--config',
                        help='''path to config file''',
                        default="./eval/inference_config.yaml",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            raise(exc)

    race_dataset = RaceDataset()

    dataset = race_dataset.get_split(config['data_split'], do_preprocess=True)

    ##TODO: Included only for testing. remove later
    #sample_test_df = test_df[test_df.groupby('context_id').ngroup() < 2]

    # create set of prompts for question generation
    race_dataset.prepare_data_qg(dataset,
                                 config['num_examples'],
                                 config['is_train'],
                                 config['is_answer_focused'])

    data_loader = torch.utils.data.DataLoader(dataset=race_dataset,
                                              batch_size=config['batch_size'],
                                              collate_fn=race_dataset.collate_fn,
                                              shuffle=False)

    print(f"LENGTH OF DATASET: {len(race_dataset)}")
    print(iter(data_loader).next()['prompt'][0])
    print(race_dataset.dataset['difficulty_label'].value_counts())
