# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:42:19 2019

@author: ceiba
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd

#def task_with_CountVectorizer():
####自带的CountVectorizer
####最终结果为0.6962770729206715
#==============================================================================
#df = pd.read_csv('./dataset/train.tsv',header=0,delimiter='\t')
#x_train = df['Phrase']
#y_train = df['Sentiment']
#count_vec = CountVectorizer()
#x_count_train = count_vec.fit_transform(x_train)
#logist = LogisticRegression()
#logist.fit(x_count_train,y_train)
#x_test = x_count_train
#predicted = logist.predict(x_test)
#print(np.mean(predicted == y_train))
## 
#==============================================================================
#task_with_CountVectorizer()

#def task_without_CountVectorizer():
####(156060,)自写CountVectorizer
####最终结果0.5221196975522235
#==============================================================================
df = pd.read_csv('./dataset/train.tsv',header=0,delimiter='\t')
x_train = df['Phrase']
y_train = df['Sentiment']
all=[]
for i in range(len(x_train)):
   all.extend(x_train[i])
voc = set(all)
x_train_idx = []
for i in range(len(x_train)):
    tmp = np.zeros(len(voc))
    for j, word in enumerate(voc):
        tmp[j] = x_train[i].count(word)	
    x_train_idx.append(tmp)
x_train_id = np.array(x_train_idx)#np.save('./data/x_train.npy',x_train_id)
#np.save('./data/y_train.npy',y_train)
#x_train = np.load('./data/x_train.npy')#y_train = np.load('./data/y_train.npy')
logist = LogisticRegression()
logist.fit(x_train_id,y_train)
x_test = x_train_id
predicted = logist.predict(x_test)
print(np.mean(predicted == y_train))
#  
#==============================================================================
 
 
 
 