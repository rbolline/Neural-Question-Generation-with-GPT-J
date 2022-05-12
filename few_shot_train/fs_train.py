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

import torch
torch.cuda.empty_cache()

import transformers
from transformers import AutoTokenizer, GPTJForCausalLM, AutoModelForCausalLM

import pandas as pd
import sklearn as sk
import yaml
import pickle
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from bitsandbytes.optim import Adam8bit

from data_preprocessing import RaceDataset
from datasets import load_dataset as hf_load_dataset

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

    train_df = race_tr_dataset.get_split('train', do_preprocess=True)
    test_df = race_te_dataset.get_split('test', do_preprocess=True)
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

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias

    def forward(self, input):
        output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None

    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"


def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)


def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

def add_adapters(model, adapter_dim=16):
    assert adapter_dim > 0

    for module in model.modules():
        if isinstance(module, FrozenBNBLinear):
            module.adapter = nn.Sequential(
                nn.Linear(module.in_features, adapter_dim, bias=False),
                nn.Linear(adapter_dim, module.out_features, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)
        elif isinstance(module, FrozenBNBEmbedding):
            module.adapter = nn.Sequential(
                nn.Embedding(module.num_embeddings, adapter_dim),
                nn.Linear(adapter_dim, module.embedding_dim, bias=False),
            )
            nn.init.zeros_(module.adapter[1].weight)

def main(config):
    """Defines main execution"""
    # load the GPT-J model
    # model = load_model(config['use_opt_model'])
    print(torch.cuda.memory_allocated())


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.padding_side = "left"

    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    #model.config.pad_token_id = model.config.eos_token_id

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

    gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
    gpt.config.pad_token_id = gpt.config.eos_token_id
    print("**** FINISHED LOADING MODEL!! *******")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt.to(device)

    param_size = 0
    for param in gpt.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in gpt.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    num_epochs = 3
    add_adapters(gpt)
    gpt.to(device)


    gpt.gradient_checkpointing_enable()

    optimizer = Adam8bit(gpt.parameters(), lr=1e-5)
    # ce_loss = nn.CrossEntropyLoss()

    with torch.cuda.amp.autocast():
       counter = 0
       for batch_dict in tqdm(train_data_loader):
            batch_prompts = batch_dict['prompt']
            batch_questions = batch_dict['question']
            print(f'\t\tTokenizing Inputs...')

            tokenized_inputs = tokenizer(batch_prompts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True).to(device)
            # pred_input_ids = tokenized_inputs.input_ids

            tokenized_labels = tokenizer(batch_questions,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True).to(device)
            # label_input_ids = tokenized_labels.input_ids


            # break
            # batch = tokenizer(row["content"], truncation=True, max_length=128, return_tensors='pt')
            # batch = {k: v.cuda() for k, v in batch.items()}

            out = gpt.forward(**tokenized_inputs,)

            loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), tokenized_inputs['input_ids'][:, 1:].flatten(),
                                reduction='mean')
            #print(loss)
            print(" **** LOSS *****", loss)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            counter += 1
            if counter == 3:
                break
            

    print("**** FINISHED TRAINING MODEL!! *******")
    # gen text from the model
    # gen_text = get_model_gen_text(model, tokenizer, test_data_loader, config)

    # return gen_text


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

    #results_df = pd.concat(results, axis=0)
    #results_df.to_csv(config['savepath'], index=False, header=True)
