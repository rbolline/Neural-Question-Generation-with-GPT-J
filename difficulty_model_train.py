import numpy as np
import pandas as pd
import torch

import transformers
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import GPT2Tokenizer, GPTJForSequenceClassification, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from datasets import load_metric

from data_preprocessing import RaceDataset
import util_methods

transformers.logging.set_verbosity_error()
# Set up GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = GPTJForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

# Generate encoded train data
train = RaceDataset('train', tokenizer)
train.preprocess_dataset()
# Generate encoded eval data
val = RaceDataset('validation', tokenizer)
val.preprocess_dataset()




num_epochs = 3
batch_size = 8
lr = 1e-5
lr_lower = 1e-5
lr_upper = 5e-5
t_args = TrainingArguments(f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}', 
                            do_train=True, 
                            num_train_epochs=num_epochs, 
                            per_device_train_batch_size=batch_size,
                            logging_dir=f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}/logs'
                        )

preds = []
eval_pred = EvalPrediction(train, train.labels)

trainer = Trainer(
    model=util_methods.model_init(),
    args=t_args,
    train_dataset=train,
    eval_dataset=val,
    compute_metrics=util_methods.compute_metrics,
    model_init=util_methods.model_init,
    tokenizer=tokenizer)

trainer.train()

eval_metrics = trainer.evaluate()
print('EVAL METRICS:\n')
# with open(f'./results/roberta_finetuned_ep={num_epochs}_batch={batch_size}/eval_metrics') as f:
#     f.write(eval_metrics.tolist())

test = RaceDataset('test', tokenizer)
test.preprocess_dataset()

pred_out = trainer.predict(test)
print('PREDICTION METRICS:\n')
# with open(f'./results/roberta_finetuned_ep={num_epochs}_batch={batch_size}/predictions') as f:
#     f.write(pred_out.tolist())
print(pred_out)

