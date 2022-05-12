
import pandas as pd
import argparse
from IPython.display import display

def extract_gen_question(row):
    query = row.loc[['pred_question']]
    ans_exp = row['answer'].replace('-', '\-').replace('.', '\.').replace('$', '\$').replace('?', '\?').replace('(', '\(').replace(')', '\)').replace(' ,', ',').strip()
    regex_exp = fr"(.+{ans_exp}.+)"

    match = query.str.extractall(regex_exp).reset_index()
    if len(match) == 0:
        print(regex_exp)
        print(query, len(match))
        return None
    else:
        return match[0].iloc[0]


def parse_new_question(row):
    query = row.loc['pred_question']
    start_idx = query.find('Question:')

    question = query[start_idx:]

    if question.find("Answer:") > 0:
        stop_idx = question.find("Answer:")
        question = question[:stop_idx]

    question = question.replace("Top<|endoftext|>", '').replace("<|endoftext|>", '')
    question = question.split('Question:')[1].strip()
    return question



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for parsing results csv")
    parser.add_argument('--input_csv',
                        help='''path to results csv''',
                        required=True)
    parser.add_argument('--output_path',
                        help='''path to processed results csv''',
                        required=True)
    args = parser.parse_args()

    results_df = pd.read_csv(args.input_csv)

    pd.set_option('max_colwidth', 500)

    results_df['context_len'] = results_df['article'].str.len()
    results_df['gen_text'] = results_df['gen_text'].str.replace(r"Top<\|endoftext\|>", '').str.replace(r"<\|endoftext\|>", '')

    results_df['pred_question'] = results_df.apply(lambda row: row['gen_text'][row['context_len'] - 1 : ], axis=1)

    # display(results_df.loc[results_df['example_id'] == 'middle6322_0', 'pred_question'].tolist())

    results_df['pred_question'] = results_df.apply(lambda row: extract_gen_question(row), axis=1)
    # display(results_df.loc[results_df['example_id'] == 'middle6322_0', 'pred_question'].tolist())

    print("AFTER EXTRACTING QUESTION")
    display(results_df[['pred_question']].head(2))

    results_df['pred_question'] = results_df.apply(lambda row: parse_new_question(row), axis=1)

    print("AFTER PARSING QUESTION")
    display(results_df[['pred_question']].head(2))

    results_df = results_df.loc[:, ["example_id", "article", "answer", "question", "options", "difficulty_label","pred_question", "gen_text", "prompt"]]

    print("BEFORE SAVING")
    display(results_df[['pred_question']].head(2))

    results_df.to_csv(args.output_path, index=False, header=True)