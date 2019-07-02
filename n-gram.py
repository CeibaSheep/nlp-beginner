# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:31:45 2019

@author: ceiba

n-gram & smooth
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
import re
from collections import Counter

N = 2

train_data = pd.read_csv('./dataset/train.tsv', header = 0, delimiter = '\t')
test_data = pd.read_csv('./dataset/test.tsv', header = 0, delimiter = '\t')

def preprocess(text):
#    text = text.lower()
    text = text.replace('.', '')
    text = text.replace('!?', '')
    text = text.replace('"', '')
    text = text.replace('#', '')
    text = text.replace('!', '')
    text = text.replace('$', '')
    text = text.replace('\'', '')
    text = text.replace('&', '')
    text = text.replace('--', '')
    text = text.replace('+', '')
    text = text.replace(',','')
    text = text.replace('-','')
    text = text.replace(':','')
    text = text.replace(';','')
    text = text.replace('=','')
    text = text.replace('?','')
    text = text.replace('\*','')
    text = text.replace('`','')
    text = text.replace('\/', '')
    text = re.sub('[0-9]+[a-zA-Z]', '', text)
    text = re.sub('\d', '', text)
    word = text.split()
    return word, text

#==============================================================================
def n_gram(words, vocabs, texts):
    dim = len(vocabs)
    vocabs = list(vocabs)
    word_count = np.zeros([dim])
    ngram_matrix = []
    for i in range(dim):
        print(i)
        word_count[i] = words.count(vocabs[i])
        
    for i in range(dim):
        tmp = np.zeros([dim])
        print(i)
        for j in range(dim):
            target = vocabs[i] + ' '+ vocabs[j]
            tmp[j] = texts.count(target)/word_count[j]
#            tmp[j] = tmp[j]/word_count[j]
        ngram_matrix.append(tmp)
        print(tmp)
    return ngram_matrix
#==============================================================================

def read_data():
    x_train = train_data['Phrase']
    y_train = train_data['Sentiment']
#    x_test = test_data['Phrase']  no label in test data
#    y_test = test_data['Sentiment']
    words = []
    texts = []
    vocabs = []
    for i in range(len(x_train)):
        word, text = preprocess(x_train[i])
        words += word
        texts.append(text)
    vocabs = set(words)
    return words, vocabs, texts

words, vocabs, texts = read_data()
str = ','
texts = str.join(texts)
ngram_matrix = n_gram(words, vocabs, texts)

    


    
    
    



