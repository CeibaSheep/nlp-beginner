# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression


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
    return words, pure_text
        
#def read_data():
td = pd.read_csv('./dataset/train.tsv', header=0, delimiter='\t')
x_train = td['Phrase']
y_train = td['Sentiment']
x_train_pure = []
words = []
pure_text = []
for item in x_train:
    word, pure_text = preprocess(item)
    words += word
    x_train_pure.append(pure_text)
vocabs = set(words)
vocab_to_int = {word: i for i, word in enumerate(vocabs)}
row_dim = len(x_train)
col_dim = len(vocabs)

batch_size = 100
def get_batch(st_idx):
    x_train_batch = x_train_pure[st_idx:(st_idx+batch_size)]
    batch_matrix = []
    for i in range(batch_size):
        tmp = np.zeros(col_dim)
        for j, word in enumerate(vocabs):
            tmp[j] = x_train_batch[i].count(word)
        batch_matrix.append(tmp)
    return batch_matrix

#def sigmoid(x):
#    return 1/(1 + exp(-x))
#    
#class LogistRegression():
#    def __init__(self, lr = .1, itera = 1000):
#        self.lr = lr
#        self.itera = itera
#        
#    def initialization(self, n_feature):
#        limit = np.sqrt(1/n_features)
#        w = np.random.uniform(-limit, limit, (n_feature, 1))
#        b = 0
#        self.w = np.insert(w, 0, b, axis = 0)
#    
#    def fit(self, x, y):
#        m_samples, n_features = X, shape
#        self.initialization(n_features)
#        X = np.insert(x, 0, 1, axis = 1)
#        y = np.reshape(y, (m_samples, 1))
#        
#        for i in range(self.n_itera):
#            h_x = X.dot(self.w)
##    
## get the vocab matrix
vocab_matrix = []
batch_len = np.int(row_dim/batch_size)
logist = LogisticRegression()
predict_arr = []
for i in range(batch_len):
    batch_matrix = get_batch(i*batch_size)
    train_matrix = np.array(batch_matrix)
    label_matrix = y_train[i*batch_size:i*batch_size+batch_size]    
    logist.fit(train_matrix, label_matrix)
    x_test = train_matrix
    predict = logist.predict(x_test)
#    predict_arr.append(predict)
    predict_arr.append(np.mean(predict == label_matrix))
#    predict.append(logist.predict(x_test))
# logistic regression
    
    
    
    
    
    
    
    
    
    

