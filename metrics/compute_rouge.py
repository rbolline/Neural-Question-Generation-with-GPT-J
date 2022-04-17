# coding=utf-8
'''Computes the ROUGE-L Score

Usage:
python rouge.py --input_csv=/path/to/input/csv --use_stemmer=true
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd

from rouge_score import rouge_scorer


def load_text(csv_path):
    """Loads the ref/hyp from the input csv"""
    df = pd.read_csv(csv_path)
    return df



def compute_rouge_score_helper(ref, hyp, scorer):
    '''Defines the ROUGE-L operation for each row in input_csv

    Args:
        ref: str. Reference text against which hypothesis is compared
        hyp: str. Hypothesis text to be evaluated

    Returns:
        precision: float. num of overlapping words / num of words in hyp text
        recall: float. num of overlapping words / num of words in ref text
        fscore: float: Harmonic mean of precision and recall
    '''

    score = scorer.score(ref, hyp)
    precision, recall, fscore = score['rougeL']
    return precision, recall, fscore


def compute_rouge_score(input_df, use_stemmer):
    '''Computes the ROUGE-L score

    Args:
        input_df: pd.DataFrame. DataFrame which contains the hypothesis
            and reference cols
        use_stemmer: bool. Flag to control if stemming should be perfomed when
            computing rouge-L score

    Returns:
        score: float. Weighted bleu_score value b/w reference and hypothesis
    '''

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    input_df['precision'], input_df['recall'], input_df['fscore'] = \
        zip(*input_df.apply(lambda row: compute_rouge_score_helper(row['reference_text'],
                                                                   row['hypothesis_text'],
                                                                   scorer),
                            axis=1)
            )

    return input_df


def main(args):
    '''Main Execution'''

    # load the input csv containing the reference and hypothesis text
    input_df = load_text(args.input_csv)

    if args.use_stemmer == 'True':
        use_stemmer = True
    else:
        use_stemmer = False

    # compute rouge-L score
    input_df = compute_rouge_score(input_df, use_stemmer)

    return input_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for BLEU score computation")

    parser.add_argument('--input_csv', help='''path to input csv''',
                        default=None,
                        required=True)
    parser.add_argument('--use_stemmer',
                        help='''flag specifying if stemming
                            should be performed during rougeL computation''',
                        default='false',
                        choices=['True', 'False'],
                        required=False)

    args = parser.parse_args()

    input_df = main(args)

    rouge_fscore = input_df['fscore'].mean()
    rouge_recall = input_df['recall'].mean()
    rouge_precision = input_df['precision'].mean()

    print("*" * 40)
    print(f"ROUGE-L F-SCORE: {rouge_fscore}\nROUGE-L PRECISION: {rouge_precision}\nROUGE-L RECALL: {rouge_recall}")
    print("-|-" * 15)
