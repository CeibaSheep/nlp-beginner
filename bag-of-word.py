# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#def read_data():
td = pd.read_csv('./dataset/train.tsv', header=0, delimiter='\t')
x_train = td['Phrase']
y_train = td['Sentiment']
words = []
vocabs = []
for item in x_train:
    words += item.split(' ')
#print('done')
#vocabs = set(words)
#x_dim = len(vocabs)
#vocab_to_int = {word: i for i, word in vocabs}
#int_to_vocab = {i: word for i, word in vocabs}
