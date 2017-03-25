
# coding: utf-8

# In[2]:

#!/usr/bin/env python2

"""
Created on Sat Mar 19 15:07:01 2017
@author: sueyang
"""


# Recall: ROUGE-N
# see formula in session 3.1 - http://83.212.103.151/~mkalochristianakis/techNotes/ipromo/rougen5.pdf 
def rouge_n(pred, refs, n = 1):
    rouge_score = []
    m = len(refs)
    pred_grams = get_ngrams(pred, n)
    
    for i in range(m):
        ref_grams = get_ngrams(refs[i], n)
        match = count_match(pred_grams, ref_grams)
        rouge_score.append(float(match) / len(ref_grams))
    return max(rouge_score)



# Precision: BLEU
def bleu(pred, refs, n = 1):
    m = len(ref)
    bleu_score = 0.0
    pred_grams = get_ngrams(pred, n)
    pred_grams_count = len(pred_grams)
    ngram_ref_list = [] # list of ngrams for each reference sentence
    for i in range(m): 
        ngram_ref_list.append(get_ngrams(refs[i], n))
  
    total_clip_count = 0
    for tuple in set(pred_grams):
        pred_count = count_element(pred_grams, tuple)
        max_ref_count = 0 
        for i in range(m):
            num = count_element(ngram_ref_list[i], tuple)
            max_ref_count = num if max_ref_count < num else max_ref_count 
        total_clip_count += min(pred_count, max_ref_count)  
  
    bleu_score = float(total_clip_count)/pred_grams_count
    return bleu_score

def count_element(list, element):
    if element in list:
        return list.count(element)
    else:
        return 0


# F1 = 2 * (Bleu * Rouge) / (Bleu + Rouge)
# https://en.wikipedia.org/wiki/F1_score
# http://stackoverflow.com/questions/38045290/text-summarization-evaluation-bleu-vs-rouge
def f1(rouge, bleu):
    return 2 * (rouge * bleu) / (bleu + rouge)



def get_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def count_match(list_1, list_2):
    match_items = []
    for item in list_1:
        if item in list_2:
            match_items.append(item)
    return len(match_items)   




