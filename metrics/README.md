Repo contains
- BLEU Score Computation (Bilingual Language Evaluation Understudy)
  - Single score for a corpus of docs.
  - Each doc is considered to be a single sentence and the entire corpus is saved as a csv file with column `text`
- ROUGE-L Score Computation (Recall-Oriented Understudy for Gisting Evaluation)
  - Using python package hosted on pip. Source code available here https://github.com/google-research/google-research/tree/master/rouge
  - Explanation of ROUGE-L score computation: https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/notes_slides/What-is-ROUGE.pdf
- BLEURT score (Bilingual Evaluation Understudy with Representations from Transformers)
  - It follows the following repo https://github.com/google-research/bleurt