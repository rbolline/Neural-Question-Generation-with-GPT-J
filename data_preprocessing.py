import numpy as np
import pandas as pd
from datasets import load_dataset


def preprocess_dataset(dataset):
    # Convert from pyarrow to pandas df
    df = dataset.to_pandas()
    # Map letter answers to options list indices
    df['answer'] = df['answer'].map({'A':0, 'B':1, 'C':2, 'D':3})
    return df

race = load_dataset('race', 'all')
df = preprocess_dataset(race['validation'])


