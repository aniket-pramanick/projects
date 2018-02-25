# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:34:10 2018

@author: aniket
"""
#Bigram Model has been used.
#Kneser-Ney Interpolated Smoothing has been used.

import nltk, math, random
from nltk.corpus import gutenberg, brown
from nltk import ngrams
from collections import Counter
from sklearn.model_selection import train_test_split

#gutenberg_sent = gutenberg.sents()
#brown_sent = brown.sents()

def padding(sent_list):
    padded_data = []
    for line in sent_list:
        #print line
        padded_data.append(['<s>']+line+['<\s>'])
    return padded_data

def divide_data(data):
    train_data, test_data = train_test_split(data, shuffle = True, test_size = 0.75)
    return (train_data, test_data)

def ngram_count(train_data):
    word_list = []   
    min_occur_words = []
    unigram_count_tmp = Counter()
    unigram_dict = Counter()
    for sentence in train_data:
        for word in sentence:
            unigram_count_tmp[word] = unigram_count_tmp[word] + 1
    
    for word in unigram_count_tmp.keys():
        if unigram_count_tmp[word] == 1:
            min_occur_words.append(word)
    rand_min_occur_words = random.sample(min_occur_words, int(len(min_occur_words)/5))
    for sentence in train_data:
        for word in sentence:
            if word in rand_min_occur_words:
                unigram_dict['<UNKN>'] += 1
                word_list.append('<UNKN>')
            else:
                unigram_dict[word] += 1
                word_list.append(word)
                
    bigram_dict = Counter(ngrams(word_list, 2))
    del bigram_dict[('<\s>','<s>')]
    #trigram_dict = Counter(ngrams(word_list, 3))
    return (unigram_dict, bigram_dict)


def test_ngram_gen(test_data, unigram_dict):
    word_list = []
    for sentence in test_data:
        for word in sentence:
            if word in unigram_dict.keys():
                word_list.append(word)
            else:
                word_list.append('<UNKN>')
    unigram_dict = Counter(word_list)
    bigram_dict = Counter(ngrams(word_list, 2))
    del bigram_dict[('<\s>', '<s>')]
    #trigram_dict = Counter(ngrams(word_list, 3))
    return (unigram_dict, bigram_dict)


def p_continuation(unigram_dict, bigram_dict):
    count_vw = Counter()
    #count_bi_w = Counter()
    count_w_w = Counter()
    lamb = Counter()
    #lamb_3 = Counter()
    for bigram in bigram_dict.keys():
        count_vw[bigram[1]] += 1
        count_w_w[bigram[0]] += 1
        d = 0.75 #Given in book
    for word in unigram_dict.keys():
        lamb[word] = (float(d)*float(count_w_w[word]))/float(unigram_dict[word])
    #count_u_w_ = sum(count_vw.values())
    return (lamb, count_vw)

    
def bigram_perplexity(test_data, train_data):
    bi_perp = 1
    prob_w = Counter()
    unigram_dict, bigram_dict = ngram_count(train_data)
    unigram_test_dict, bigram_test_dict = test_ngram_gen(test_data, unigram_dict)
    lamb, count_vw = p_continuation(unigram_dict, bigram_dict)
    count_u_w_ = sum(count_vw.values())    
    t = sum(bigram_test_dict.values())
    for bigram in bigram_test_dict.keys():      
        first = bigram[0]
        second = bigram[1]
        freq = bigram_test_dict[bigram]
        prob_w[bigram] = (max(((bigram_dict[bigram])-0.75), 0)/unigram_dict[first]) + (lamb[first]*(float(count_vw[second])/count_u_w_))
        if prob_w[bigram]>0:
            bi_perp = bi_perp * pow(1.0/prob_w[bigram], freq/t)
        
    return bi_perp
    
def gen_tokens():
    pass


def main():
    padded_data = padding(gutenberg.sents())
    train_data, test_data = divide_data(padded_data)
    perp = bigram_perplexity(test_data, train_data)
    print perp


if __name__ == '__main__':
    main()
    