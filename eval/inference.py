# coding=utf-8
'''Runs inference using GPT-J for Question Generation

Usage:
python inference.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.abspath("../"))

import argparse
import pandas as pd
from tqdm import tqdm
import yaml
import pickle

import torch
from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM

from data_preprocessing import RaceDataset


def load_model(use_opt_model=True):
    """Loads GPT-J model for inference

    Args:
        opt_model: bool. If an optimized version of the model should be loaded

    """
    if use_opt_model:
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",
                                                revision="float16",
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

    return model


def load_dataset(config):
    race_dataset = RaceDataset()

    test_df = race_dataset.get_split('test', do_preprocess=True)
    print(test_df.columns)
    print(test_df.head(1))

    ##TODO: Included only for testing. remove later
    sample_test_df = test_df[test_df.groupby('example_id').ngroup() < 2]

    # create set of prompts for question generation
    prompt_df = race_dataset.prepare_data_qg(sample_test_df,
                                             config['num_examples'],
                                             config['is_train'])

    # batch the inputs for infernce
    batch_size = config['batch_size']
    batch_prompts = [prompt_df.iloc[idx : idx + batch_size] for idx in range(0, len(prompt_df), batch_size)]

    return batch_prompts


def get_model_gen_text(model,
                       tokenizer,
                       batched_prompt_df,
                       config):
    """Generates text using model and prompt"""
    # if cuda exists, use cuda, else run on cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    model.to(device)

    model_params = config['model_params']
    results = []
    for batch_df in tqdm(batched_prompt_df):
        batch_prompts = batch_df['prompt'].tolist()
        print(batch_prompts)

        tokenized_inputs = tokenizer(batch_prompts,
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True).to(device)
        input_ids = tokenized_inputs.input_ids

        print(input_ids)

        gen_tokens = model.generate(input_ids, **model_params)
        gen_text = tokenizer.batch_decode(gen_tokens)

        batch_df['gen_text'] = gen_text
        results.append(batch_df)

    return results


def main(config):
    """Defines main execution"""
    # load the GPT-J model
    # model = load_model(config['use_opt_model'])

    print("**** FINISHED LOADING MODEL!! *******")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = model.config.eos_token_id

    print("**** FINISHED LOADING TOKENIZER!! *******")

    # prepare the dataset of prompts
    batched_prompt_df = load_dataset(config)

    print("**** FINISHED LOADING DATSET!! *******")
    print(batched_prompt_df[0].head(2))

    # gen text from the model
    gen_text = get_model_gen_text(model, tokenizer, batched_prompt_df, config)

    return gen_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")
    parser.add_argument('--config',
                        help='''path to config file''',
                        default="./inference_config.yaml",
                        required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            raise(exc)

    results = main(config)

    # TODO: change this later. save results
    with open(r'./sample_output.pickle', 'wb') as fh:
        pickle.dump(results, fh)

    results_df = pd.concat(results, axis=0)
    results_df.to_csv(config['savepath'], index=False, header=True)
