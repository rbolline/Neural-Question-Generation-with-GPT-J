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
import traceback
import pandas as pd
from tqdm import tqdm
import yaml
import pickle

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

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

    test_df = race_dataset.get_split(config['data_split'], do_preprocess=True)

    ##TODO: Included only for testing. remove later
    test_df = test_df[test_df.groupby('context_id').ngroup() < 2]

    # create set of prompts for question generation
    race_dataset.prepare_data_qg(test_df,
                                 config['num_examples'],
                                 config['is_train'],
                                 config['is_answer_focused'])

    return race_dataset


def get_model_gen_text(model,
                       tokenizer,
                       data_loader,
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
    try:
        counter = 0
        for batch_dict in tqdm(data_loader):
            try:
                batch_prompts = batch_dict['prompt']

                tokenized_inputs = tokenizer(batch_prompts,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True).to(device)
                input_ids = tokenized_inputs.input_ids

                # print("AFTER INPUT IDS")
                # print(torch.cuda.memory_allocated())

                gen_tokens = model.generate(input_ids, **model_params)
                gen_text = tokenizer.batch_decode(gen_tokens)

                # send the tokenized inputs back to the cpu
                del tokenized_inputs
                del input_ids
                torch.cuda.empty_cache()

                # print("AFTER DELETING CACHE")
                # print(torch.cuda.memory_allocated())

                batch_dict['gen_text'] = gen_text

                batch_df = pd.DataFrame(batch_dict)
                results.append(batch_df)

                counter += 1

                if counter % 50 == 0:
                    print("AFTER 50 examples")
                    print(torch.cuda.memory_allocated())

                    results_df = pd.concat(results, axis=0)
                    results_df.to_csv(config['savepath'], index=False, header=True)

            except Exception as err:
                raise(err)
                traceback.print_exc()

    except KeyboardInterrupt as err:
        print("Exiting due to Keyboard Interrupt")

    return results


def main(config):
    """Defines main execution"""
    # load the GPT-J model
    model = load_model(config['use_opt_model'])
    print(torch.cuda.memory_allocated())


    print("**** FINISHED LOADING MODEL!! *******")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    print("**** FINISHED LOADING TOKENIZER!! *******")

    # prepare the dataset of prompts
    test_dataset = load_dataset(config)

    print("**** FINISHED LOADING DATASET!! *******")
    print(test_dataset[0])

    print(len(test_dataset.dataset))

    data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config['batch_size'],
                                              collate_fn=test_dataset.collate_fn,
                                              shuffle=False)

    # gen text from the model
    # model=None
    gen_text = get_model_gen_text(model, tokenizer, data_loader, config)

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

    results_df = pd.concat(results, axis=0)
    results_df.to_csv(config['savepath'], index=False, header=True)
