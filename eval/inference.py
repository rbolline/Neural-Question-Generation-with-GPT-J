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


def load_dataset(num_examples, is_train):
    race_dataset = RaceDataset()

    test_df = race_dataset.get_split('test', do_preprocess=True)
    print(test_df.columns)
    print(test_df.head(1))

    ##TODO: Included only for testing. remove later
    sample_test_df = test_df[test_df.groupby('example_id').ngroup() < 2]

    # create set of prompts for question generation
    prompt_collection = race_dataset.prepare_data_qg(sample_test_df, num_examples, is_train)

    return prompt_collection


def get_model_gen_text(model, tokenizer, prompt):
    """Generates text using model and prompt"""
    # if cuda exists, use cuda, else run on cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    model.to(device)

    tokenized_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized_inputs.input_ids

    gen_tokens = model.generate(input_ids,
                                do_sample=True,
                                temperature=0.9,
                                max_length=100,)

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    return gen_text


def main(args):
    """Defines main execution"""
    if args.use_opt_model == "True":
        use_opt_model = True
    else:
        use_opt_model = False

    if args.is_train == "True":
        is_train = True
    else:
        is_train = False

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    # load the GPT-J model
    model = load_model(use_opt_model)

    # prepare the dataset of prompts
    prompt_collection = load_dataset(args.num_examples, is_train)

    # gen text from the model
    gen_text = get_model_gen_text(model, tokenizer, prompt_collection)

    return gen_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")
    parser.add_argument('--use_opt_model',
                        help='''if optimized GPT-J model should be used for
                            inference. Includes fp16 precision''',
                        default="True",
                        choices=['True', 'False'],
                        required=False)
    parser.add_argument('--is_train', help='''training or inference mode''',
                        default="False",
                        choices=['True', 'False'],
                        required=False)
    parser.add_argument('--num_examples',
                        help='''num of example questions to use in
                            each prompt''',
                        default=1,
                        required=False)

    args = parser.parse_args()

    gen_text = main(args)

    # TODO: change this later. save results
    pd.DataFrame([gen_text]).to_csv("./sample_gen_text_output.csv", index=True, header=True)
