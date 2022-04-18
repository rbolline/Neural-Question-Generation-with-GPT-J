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


def load_dataset(num_examples, is_train, batch_size):
    race_dataset = RaceDataset()

    test_df = race_dataset.get_split('test', do_preprocess=True)
    print(test_df.columns)
    print(test_df.head(1))

    ##TODO: Included only for testing. remove later
    sample_test_df = test_df[test_df.groupby('example_id').ngroup() < 2]

    # create set of prompts for question generation
    prompt_collection = race_dataset.prepare_data_qg(sample_test_df, num_examples, is_train)

    # batch the inputs for infernce
    if batch_size == 1:
        batch_prompts = [[prompt] for prompt in prompt_collection]
    else:
        batch_prompts = [prompt_collection[idx : idx + batch_size] for idx in range(0, len(prompt_collection), batch_size)]

    return batch_prompts


def get_model_gen_text(model, tokenizer, prompts):
    """Generates text using model and prompt"""
    # if cuda exists, use cuda, else run on cpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    model.to(device)

    results = []
    for batch_prompts in tqdm(prompts):
        tokenized_inputs = tokenizer(batch_prompts,
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True).to(device)
        input_ids = tokenized_inputs.input_ids

        gen_tokens = model.generate(input_ids,
                                    do_sample=True,
                                    temperature=0.3,
                                    max_length=800,)

        gen_text = tokenizer.batch_decode(gen_tokens)[0]

        results.extend(gen_text)

    return results


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

    # load the GPT-J model
    model = load_model(use_opt_model)

    print("**** FINISHED LOADING MODEL!! *******")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    print("**** FINISHED LOADING TOKENIZER!! *******")

    # prepare the dataset of prompts
    batched_prompts = load_dataset(args.num_examples, is_train, int(args.batch_size))

    print("**** FINISHED LOADING DATSET!! *******")
    print(batched_prompts[0])

    # gen text from the model
    gen_text = get_model_gen_text(model, tokenizer, batched_prompts)

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
    parser.add_argument('--batch_size',
                        help='''number of examples per batch during inference''',
                        default=2,
                        required=False)
    args = parser.parse_args()

    gen_text = main(args)

    # TODO: change this later. save results
    pd.DataFrame(gen_text).to_csv("./sample_gen_text_output.csv", index=True, header=True)
