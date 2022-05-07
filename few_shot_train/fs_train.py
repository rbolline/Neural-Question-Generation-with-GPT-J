# coding=utf-8
'''Runs few shot training, then inference, using GPT-J for Question Generation.
GCP INSTRUCTIONS:
1. Log onto Greene
2. ssh burst
3. reserve GPU (usually srun --account=ds_ga_1012_2022sp --partition=n1s8-v100-1 
                --gres=gpu --time=2:00:00 --pty /bin/bash)
4. go to few_shot_train folder
5. run singularity container (usually singularity exec --nv --bind /scratch --overlay 
                                /scratch/rb4987/overlay-25GB-500K.ext3:rw 
                                /scratch/rb4987/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif 
                                /bin/bash)
6. run the command: python fs_train.py


Usage:
python fs_train.py
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
import sklearn as sk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import yaml
import pickle

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EvalPrediction

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
    race_tr_dataset = RaceDataset()
    race_val_dataset = RaceDataset()
    race_te_dataset = RaceDataset()

    train_df = race_tr_dataset.get_split(config['train_data_split'], do_preprocess=True)
    test_df = race_te_dataset.get_split(config['test_data_split'], do_preprocess=True)
    val_df = race_val_dataset.get_split('validation', do_preprocess=True)

    # TODO: Included only for testing. remove later
    train_df = train_df[train_df.groupby('context_id').ngroup() < 2]
    test_df = test_df[test_df.groupby('context_id').ngroup() < 2]
    val_df = val_df[val_df.groupby('context_id').ngroup() < 2]

    # create set of prompts for question generation
    race_tr_dataset.prepare_data_qg(train_df,
                                 config['num_examples'],
                                 config['is_train'],
                                 config['is_answer_focused'])
    race_te_dataset.prepare_data_qg(test_df,
                                 config['num_examples'],
                                 False,
                                 config['is_answer_focused'])
    race_val_dataset.prepare_data_qg(val_df,
                                 config['num_examples'],
                                 False,
                                 config['is_answer_focused'])
    return race_tr_dataset, race_val_dataset, race_te_dataset


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

# def compute_metrics(eval_pred):
#     labels = eval_pred.label_ids
#     preds = eval_pred.predictions.argmax(-1)

#     (precision, recall, f1, _) = precision_recall_fscore_support(labels, preds)
#     metrics_dict = {
#         'accuracy': accuracy_score(labels, preds),
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }
#     return metrics_dict

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
    train_dataset, val_dataset, test_dataset = load_dataset(config)

    print("**** FINISHED LOADING DATASET!! *******")

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=config['batch_size'],
                                              collate_fn=test_dataset.collate_fn,
                                              shuffle=False)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config['batch_size'],
                                              collate_fn=test_dataset.collate_fn,
                                              shuffle=False)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=config['batch_size'],
                                              collate_fn=test_dataset.collate_fn,
                                              shuffle=False)
    # Train Model
    num_epochs = 3
    t_args = TrainingArguments(config['outpath'], 
                                do_train=True, 
                                num_train_epochs=num_epochs, 
                                per_device_train_batch_size=config['batch_size'],
                                per_device_eval_batch_size=config['batch_size']
                            )
    # preds = []
    # eval_pred = EvalPrediction(train_dataset, train_dataset.dataset['question'])

    trainer = Trainer(
        args=t_args,
        train_dataset=train_data_loader,
        eval_dataset=val_data_loader,
        # compute_metrics=compute_metrics,
        # data_collator=train_dataset.collate_fn,
        model_init=load_model,
        tokenizer=tokenizer)

    trainer.train()

    print("**** FINISHED TRAINING MODEL!! *******")
    # gen text from the model
    gen_text = get_model_gen_text(model, tokenizer, test_data_loader, config)

    return gen_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")
    parser.add_argument('--config',
                        help='''path to config file''',
                        default="./fs_train_config.yaml",
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
