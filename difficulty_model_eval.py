import numpy as np
import pandas as pd
import torch
import json

import transformers
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import GPT2Tokenizer, GPTJForSequenceClassification, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from datasets import load_metric

from data_preprocessing import RaceDataset
import util_methods

num_epochs = 2
batch_size = 8


try:
    model = RobertaForSequenceClassification.from_pretrained(f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}/checkpoint-500')
    tokenizer = RobertaTokenizer.from_pretrained(f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}/checkpoint-500')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    metrics_df = pd.read_csv('./metrics_df.csv')
    dataset = metrics_df.drop(columns = ['options','hypothesis_text', 'gen_text', 'prompt'])
    # combined = 'Passage: ' + dataset['article'] + ', Question: ' + dataset['question'] + ', Answer: ' + dataset['answer']
    test = RaceDataset('test', tokenizer)
    test.test_dataset(dataset)

    t_args = TrainingArguments(f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}', 
                        do_train=False, 
                        num_train_epochs=num_epochs, 
                        per_device_train_batch_size=batch_size,
                        logging_dir=f'./results/roberta_difficulty_ep={num_epochs}_batch={batch_size}/logs'
                    )
    preds = []
    eval_pred = EvalPrediction(test, test.labels)

    trainer = Trainer(
        args=t_args,
        eval_dataset=test,
        compute_metrics=util_methods.compute_metrics,
        model_init=util_methods.model_init,
        tokenizer=tokenizer)

    eval_metrics = trainer.evaluate()
    print('EVAL METRICS:\n')
    print(eval_metrics)
    
    with open(f'./results/roberta_finetuned_ep={num_epochs}_batch={batch_size}/eval_metrics', 'w') as f:
        json.dump(eval_metrics, f)
    predict = trainer.predict(test)
    with open(f'./results/roberta_finetuned_ep={num_epochs}_batch={batch_size}/predictions', 'w') as f:
        json.dump(predict, f)
    print(predict)
except Exception as e:
    print(e)
    
