import sklearn as sk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch

from transformers import GPTJForSequenceClassification, AutoModelForCausalLM, RobertaForSequenceClassification

# Modified methods from DSGA1012 Lab 3

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    (precision, recall, f1, _) = precision_recall_fscore_support(labels, preds)
    metrics_dict = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return metrics_dict
    
def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.
    # print(dataset[['question', 'article', 'answer']].values.tolist()[:5])
    dataset['combined_context'] = dataset['article'] + dataset['question']
    # print(dataset[['combined_context', 'answer']].values.tolist()[:2])
    out = tokenizer.batch_encode_plus(dataset[['combined_context', 'answer']].values.tolist(), 
                                      max_length=max_seq_length, 
                                      truncation=True, padding=True,
                                      return_tensors='pt')
    return out

def model_init():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GPTJForSequenceClassification.from_pretrained("EleutherAI/gpt-j-6B")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    # model = model.to(device)
    # model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
    return model