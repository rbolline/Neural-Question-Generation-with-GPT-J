import numpy as np
import pandas as pd
import torch

from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import GPT2Tokenizer, GPTJForSequenceClassification, AutoModelForCausalLM
# from datasets import load_metric

from data_preprocessing import RaceDataset
import util_methods

# Set up tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = GPTJForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

# Generate encoded train data
train = RaceDataset('train')
train.preprocess_dataset()
train_df = train.dataset.iloc[:2000]
train_labels = train_df['example_id'].map(lambda x: 'high' if 'high' in x else 'middle')
train_df['label'] = train_labels
train_df = train_df.drop(columns=['example_id'])
train_data = util_methods.encode_data(train_df, tokenizer)

# Generate encoded eval data
val = RaceDataset('validation')
val.preprocess_dataset()
val_df = val.dataset.iloc[:500]
val_labels = val_df['example_id'].map(lambda x: 'high' if 'high' in x else 'middle')
val_df['label'] = val_labels
val_df = val_df.drop(columns=['example_id'])
val_data = util_methods.encode_data(val_df, tokenizer)



num_epochs = 3
batch_size = 8
lr = 1e-5
lr_lower = 1e-5
lr_upper = 5e-5
t_args = TrainingArguments('./output', 
                            do_train=True, 
                            num_train_epochs=num_epochs, 
                            per_device_train_batch_size=batch_size
                        )

preds = []
eval_pred = EvalPrediction(train_data, train_df['label'])

trainer = Trainer(
    model=util_methods.model_init,
    args=t_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=util_methods.compute_metrics,
    model_init=util_methods.model_init,
    tokenizer=tokenizer)

trainer.train()