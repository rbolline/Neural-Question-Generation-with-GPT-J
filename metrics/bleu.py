# coding=utf-8
'''Computes the Cumulative BLEU-1 to BLEU-4 scores

Usage:
python bleu.py --weights=0.25,0.25,0.25,0.25 \
               --reference=/path/to/csv \
               --hypothesis=/path/to/csv
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu


def load_text(csv_path):
    """Loads the ref/hyp from the input csv"""
    df = pd.read_csv(csv_path)
    text = df['text'].str.split().values.tolist()
    return text


def compute_corpus_blue(ref, hyp, bleu_weights):
    '''Computes the corpus bleu score

    Args:
        ref: list[list[str]]. List of list. Each inner list contains
            tokens of reference text
        hyp: list[list[str]]. List of list. each inner list contains
            tokens of hypothesis text
        blue_weights: tuple[float]. Weights to be assigned to BLEU-1 to BLEU-4
            when computing the bleu_score

    Returns:
        score: float. Weighted bleu_score value b/w reference and hypothesis
    '''
    score = corpus_bleu(ref, hyp, bleu_weights)
    return score


def main(args):
    '''Main Execution'''

    # load the input reference and hypothesis csvs
    reference = load_text(args.reference)
    hypothesis = load_text(args.hypothesis)

    # parse the bleu_score weights. Must be passed as tuple of floats
    bleu_weights = args.weights.split(",")
    bleu_weights = tuple([float(x) for x in bleu_weights])

    if sum(bleu_weights) != 1.0:
        raise ValueError(f"The bleu weights provided must sum to 1. Sum={sum(bleu_weights)}")

    score = compute_corpus_blue(reference, hypothesis, bleu_weights)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")

    parser.add_argument('--reference', help='''path to reference csv''',
                        default=None,
                        required=True)
    parser.add_argument('--hypothesis', help='''path to hypothesis csv''',
                        default=None,
                        required=True)
    parser.add_argument('--weights', help='''bleu score weights. Must be comma
                        separated string of values''',
                        default='0.25, 0.25, 0.25, 0.25',
                        required=False)

    args = parser.parse_args()

    bleu_score = main(args)

    print("*" * 40)
    print(f"BLEU SCORE: {bleu_score}")
    print("-|-" * 15)