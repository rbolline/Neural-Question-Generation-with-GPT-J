# coding=utf-8
'''Computes the BLEURT Score

Usage:
python bleurt.py --input_csv=/path/to/input/csv \
                 --model_name=BLEURT-20
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from datasets import load_metric


def load_text(csv_path):
    """Loads the ref/hyp from the input csv"""
    df = pd.read_csv(csv_path)
    reference_texts = df['reference_text'].values.tolist()
    hypothesis_texts = df['hypothesis_text'].values.tolist()

    return reference_texts, hypothesis_texts


def compute_bleurt_score(ref, hyp, model_name):
    '''Computes the bleurt score

    Args:
        ref: list[str]. List of reference text strings.
        hyp: list[str]. List of hypothesis text strings
        model_name: str. Model checkpoint to use for BLEURT score computation

    Returns:
        scores: dict. List of BLEURT scores for each (ref, hyp) input pair
    '''
    bleurt = load_metric("bleurt", model_name)
    scores = bleurt.compute(ref, hyp)

    return scores


def main(args):
    '''Main Execution'''

    # load the input csv containing the reference and hypothesis text
    reference_texts, hypothesis_texts = load_text(args.input_csv)

    # compute the BLEURT score
    scores = compute_bleurt_score(reference_texts, hypothesis_texts, args.model_name)
    bleurt_score = np.mean(scores['scores'])

    return bleurt_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")

    parser.add_argument('--input_csv', help='''path to input csv''',
                        default=None,
                        required=True)
    parser.add_argument('--model_name',
                        help='''Model checkpoint to use for
                            BLEURT score computation''',
                        default='BLEURT-20',
                        required=False)

    args = parser.parse_args()

    bleurt_score = main(args)

    print("*" * 40)
    print(f"BLEURT SCORE: {bleurt_score}")
    print("-|-" * 15)
