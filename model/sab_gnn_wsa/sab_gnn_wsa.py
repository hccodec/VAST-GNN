#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import torch.nn.functional as F

# In[3]:


class SpecGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5, concat=True):
        super(SpecGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(input_dim, hidden_dim)))
        self.W.requires_grad = True
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
    
    def forward(self, feature, D_n_A_D_n):
        #feature.shape: (N,input_dim)
        feature_new = torch.mm(feature.float(), self.W)
        feature_new  = F.dropout(feature_new, self.dropout, training=self.training)
        H = torch.mm(D_n_A_D_n.float(), feature_new)
        return H
        
class SpecGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.5):
        super(SpecGCN, self).__init__()
        self.dropout = dropout
        #self.layer1 = SpecGCNLayer(input_dim, hidden_dim, dropout=dropout, concat=True)
        #self.out_feature = SpecGCNLayer(hidden_dim, out_dim, dropout=dropout, concat=True)
        self.layer1 = SpecGCNLayer(input_dim, out_dim, dropout=dropout, concat=True)
        
    def compute_D_n_A_D_n(self, adjs):
        N =  adjs.size()[0]   
        tilde_A = adjs + torch.eye(N)
        tilde_D_n = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        D_n_A_D_n = torch.mm(tilde_D_n, torch.mm(tilde_A, tilde_D_n))
        return D_n_A_D_n 
    
    def forward(self, x, adjs):
        D_n_A_D_n = self.compute_D_n_A_D_n(adjs)
        #x1 = self.layer1(x, D_n_A_D_n)
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        #x2 = self.out_feature(x1, D_n_A_D_n)
        #x_out = F.relu(x2)   
        x1 = self.layer1(x, D_n_A_D_n)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x_out = F.relu(x1)
        return x_out
    
class SpecGCN_LSTM(nn.Module):
    def __init__(self, x_days, y_days, input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2, dropout_1, N):
        super (SpecGCN_LSTM, self).__init__()
        self.N = N
        self.x_days = x_days
        self.y_days = y_days
        self.specGCN = SpecGCN(input_dim_1, hidden_dim_1, out_dim_1, dropout_1)
        
        self.lstm = nn.LSTM(batch_first=True, input_size=out_dim_1+1, hidden_size=hidden_dim_2,num_layers=1, bidirectional=False)
        
        #self.lstm = nn.LSTM(batch_first=True, input_size=N*out_dim_1+N, hidden_size=N*hidden_dim_2,num_layers=1, bidirectional=False)     
        self.out_dim_1 = out_dim_1             #out dimension of GNN
        self.hidden_dim_2 = hidden_dim_2       #out dimension of LSTM
        self.fc1 = nn.Linear(self.hidden_dim_2 + self.x_days, 1)
        self.v = torch.nn.Parameter(torch.empty(23))
        self.v.requires_grad = True
        torch.nn.init.normal_(self.v.data, mean=0.005, std=0.000)
        
    def run_specGCN_lstm(self, input_record):
        N = self.N
        n_batch = len(input_record)
        x_days, y_days = round(len(input_record[0][0])/2), len(input_record[0][1])
        
        #Step1: calculate the SpecGCN output        
        #lstm_input_batch = torch.zeros((N, n_batch, self.x_days, self.out_dim_1+1))   
        for batch in range(n_batch):
            #mobility
            #torch.tensor(input_record[batch][0][2*i]).float()     #(N,N)
            #text
            #torch.tensor(input_record[batch][0][2*i+1]).float()   #(N,text_dimension)
            #infection
            #torch.tensor(input_record[batch][2][i]).float()       #N
            lstm_input_batch = torch.zeros((N, n_batch, self.x_days, self.out_dim_1+1))
            for i in range(x_days):
                x  = torch.tensor(input_record[batch][0][2*i+1]).float()   #(N, text_dimension)
                adj = torch.tensor(input_record[batch][0][2*i]).float()    #(N, N)
                x_infection = torch.tensor(input_record[batch][2][i]).float() #N
                x_infection = x_infection.reshape((x_infection.size()[0],1)) #(N, 1)
                day_order =  input_record[batch][3]                           #v6
                specGCN_out = self.specGCN(x, adj)                        #(N, out_dim1)                 
                specGCN_out = torch.cat([specGCN_out, x_infection], dim=1) #(N, out_dim1+1) 
                for j in range(N):
                    lstm_input_batch[j][batch][i] = specGCN_out[j]    
        #lstm_input_batch.shape = [N, batch, x_days, out_dim_1+1]
        
        #Step2: calculate the LSTM outputs
        y_output_batch = torch.zeros((n_batch, self.y_days, self.N))
        for j in range(N):
            lstm_input_x1 = lstm_input_batch[j]                   #[batch, x_days, out_dim_1+1]
            lstm_input_x2 = torch.mean(lstm_input_x1, dim=1)
            lstm_input_x2 = lstm_input_x2.view((lstm_input_x2.size()[0], 1, lstm_input_x2.size()[1])) ##[batch, 1, out_dim_1+1]
            lstm_input_x2 = lstm_input_x2.repeat(1, y_days, 1)  ##[batch, y_days, out_dim_1+1]
            
            lstm_input_x1_output, (hc,cn) = self.lstm(lstm_input_x1)          #[batch, x_days, hidden_dim_2]
            lstm_output, (hc1,cn1) = self.lstm(lstm_input_x2, (hc,cn))        #[batch, y_days, hidden_dim_2]
            
            infection_tensor = torch.tensor([[input_record[batch][2][x_day_c][j] for x_day_c in range(x_days)] for batch in range(n_batch)]).float()
            infection_tensor = infection_tensor.view((infection_tensor.size()[0], 1, infection_tensor.size()[1]))
            infection_tensor = infection_tensor.repeat(1, y_days, 1)  #[batch, y_days, x_days]
            
            lstm_output = torch.cat([lstm_output, infection_tensor], dim=2)
            
            
            fc_output = self.fc1(lstm_output)   #[batch, y_days, 1]
            fc_output = F.relu(fc_output)    #[batch, y_days, 1]
            if j == 0:
                y_output_batch = fc_output
            else:
                y_output_batch = torch.cat([y_output_batch, fc_output], dim=2)
        return y_output_batch      #y_output_batch.shape = (batch, y_day, N)


# In[4]:






