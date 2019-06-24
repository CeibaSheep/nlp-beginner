# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:03:59 2019
logistic/softmax regression for text classification
duration: 2 weeks
@author: ceiba
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import random
#from sklearn.feature_extaction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression

def data_process():
    td = pd.read_csv('./dataset/train.tsv', header = 0, delimiter='\t')
    x_train = td['Phrase']
    y_train = td['Sentiment']
    all = []
    for i in range(len(x_train)):
        all.extend(x_train[i]) # all chars of x_train
    charset = set(all) # no order 
    print(charset)
    word_to_idx = {word: i for i, word in enumerate(charset)} # construct dict
    g_x_train = []
    g_y_train = []
    for i in range(len(x_train)):
        sen_temp = [word_to_idx[i] for i in x_train[i]] # ordered index of char
        len_temp = len(sen_temp)
        if len_temp > 2:
            for j in range(len_temp-2):
                g_x_train.append((sen_temp[j], sen_temp[j+1]))
                g_y_train.append(sen_temp[j+2])
    np.save('./dataset/n_gram_x.npy', g_x_train)
    np.save('./dataset/n_gram_y.npy', g_y_train)

data_process()
x_all = np.array(np.load('./dataset/n_gram_x.npy'))
y_all = np.array(np.load('./dataset/n_gram_y.npy'))
lenth = len(y_all)
voc_size = 80
torch.manual_seed(1)
epoch = 10
em_dim = 100
pre_len = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mlp(nn.Module):
    def __init__(self, voc_size, em_dim, pre_len):
        super(mlp, self).__init__() # super().__init__
        self.em = torch.nn.Embedding(voc_size, em_dim)
        self.fc1 = torch.nn.Linear(em_dim, 128) #?
        self.fc2 = torch.nn.Linear(128, voc_size)
        
    def forward(self, din):
        dout = self.em(din)
        dout = torch.nn.functional.relu(self.fc1(dout))
        dout = torch.sum(dout, 1)
        if self.training:
            dout = torch.nn.functional.softmax(torch.nn.functional.relu(self.fc2(dout)))
        return dout
    
model = mlp(voc_size, em_dim, pre_len).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
batch_size = 75
itor = int(lenth/batch_size)

def train(model, device, optimizer, epoch):
    # model.train()
    for i in range(itor):
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            batch_x.append(x_all[i*batch_size+j])
            batch_y.append(y_all[i*batch_size+j])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        data1 = torch.tensor(batch_x, dtype = torch.long).to(device)
        target = torch.tensor(batch_y, dtype = torch.long).to(device)
        optimizer.zero_grad()
        output = model(data1)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (i+1)%1000 == 0:
            print('train epoch:{}\tLoss:{:.6f}'.format(epoch, loss.item()))

def test(model, device):
    correct = 0
    number = 100*batch_size
    with torch.no_grad():
        for i in range(100):
            batch_x = []
            batch_y = []
            for j in range(batch_size):
                batch_x.append(x_all[i*batch_size+j])
                batch_y.append(y_all[i*batch_size+j])
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            data1 = torch.tensor(batch_x, dtype=torch.long).to(device)
            target = torch.tensor(batch_y, dtype=torch.long).to(device)
            output = model1(data1)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('Test set: AccurarcyL {}/{}({:.0f}%)\n'.format(correct, number, 100.*correct/number))

#def output(model, device):
#    model.load_state_dict(torch.load)
total = sum([param.nelement() for param in model.parameters()])
for epoch in range(1, epoch+1):
    train(model, DEVICE, optimizer, epoch)
#    test(model, DEVICE)l











