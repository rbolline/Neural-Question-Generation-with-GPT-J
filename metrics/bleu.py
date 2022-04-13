# coding=utf-8
'''Computes the Cumulative BLEU-1 to BLEU-4 scores

Usage:
python bleu.py --run_cumulative=true \
               --weights=0.25,0.25,0.25,0.25 \
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
    '''Computes the corpus bleu score'''
    score = corpus_bleu(ref, hyp, bleu_weights)
    return score


def main(args):
    '''Main Execution'''

    reference = load_text(args.reference)
    hypothesis = load_text(args.hypothesis)

    scores = {}
    if args.run_cumulative == 'true':
        cum_bleu_weights = [(1, 0, 0, 0),
                            (0.5, 0.5, 0, 0),
                            (0.33, 0.33, 0.33, 0),
                            (0.25, 0.25, 0.25, 0.25)]

        for idx, bleu_weights in enumerate(cum_bleu_weights, start=1):
            score = compute_corpus_blue(reference, hypothesis, bleu_weights)
            scores[f'BLEU-{idx}'] = score
    else:
        # parse the bleu_score weights. Must be passed as tuple of floats
        bleu_weights = args.weights.split(",")
        bleu_weights = (float(x) for x in bleu_weights)

        if sum(bleu_weights) != 1.0:
            raise ValueError(f"The bleu weights provided must sum to 1. Sum={sum(bleu_weights)}")

        score = compute_corpus_blue(reference, hypothesis, bleu_weights)
        scores['BLEU'] = score

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")

    parser.add_argument('--run_cumulative', help='''specify if cumulative BLEU-1
                        to BLEU-4 should be computed''',
                        default='false',
                        choices=['true', 'false'])
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

    print("*" * 50)
    print(f"SUMMARY OF BLEU METRICS")
    print("*" * 50)
    for k, v in bleu_score.items():
        print(f"{k}: {v}")

    print("-|-" * 20)