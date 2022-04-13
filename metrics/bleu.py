# coding=utf-8
'''Computes the Cumulative BLEU-1 to BLEU-4 scores

Usage:
python bleu.py

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu


def load_text(csv_path, mode):
    """Loads the ref/hyp from the input csv"""
    df = pd.read_csv(csv_path)
    text = df['text'].str.split().values.tolist()
    print(text)
    return text


def compute_sentence_blue(ref, hyp, bleu_weights):
    '''Computes the sentence bleu score'''
    score = sentence_bleu(ref, hyp, bleu_weights)
    return score

def compute_corpus_blue(ref, hyp, bleu_weights):
    '''Computes the corpus bleu score'''
    score = corpus_bleu(ref, hyp, bleu_weights)
    return score


def main(args):
    '''Main Execution'''

    # parse the bleu_score weights. Must be passed as tuple of floats
    bleu_weights = args.weights.split(",")
    bleu_weights = (float(x) for x in bleu_weights)

    reference = load_text(args.reference)
    hypothesis = load_text(args.hypothesis)

    if args.mode == 'sentence':
        score = compute_sentence_blue(reference, hypothesis, bleu_weights)

    else:
        score = compute_corpus_blue(reference, hypothesis, bleu_weights)

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")

    parser.add_argument('--mode', help='''specify if the input is in
                        the form of documents or individual sentences''',
                        default='sentence',
                        choices=['sentence', 'corpus'])
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