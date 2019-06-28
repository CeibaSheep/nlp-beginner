# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re


def preprocess(text):
    text = text.lower()
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
    words = text.split()
    return words
        
#def read_data():
td = pd.read_csv('./dataset/train.tsv', header=0, delimiter='\t')
x_train = td['Phrase']
y_train = td['Sentiment']
words = []
for item in x_train:
    words += preprocess(item)
vocabs = set(words)
vocab_to_int = {word: i for i, word in enumerate(vocabs)}
row_dim = len(x_train)
col_dim = len(vocabs)


vocab_matrix = []
for i in range(row_dim):
    word_list = list(set(x_train[i].split()))
    tmp = np.zeros(col_dim)
    for j, word in enumerate(vocabs):
        tmp[j] = x_train[i].count(word)
    vocab_matrix.append(tmp)

