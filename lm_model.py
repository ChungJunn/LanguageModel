from __future__ import unicode_literals, print_function, division

import math
import numpy as np
import random
import copy

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from mylib.layers import CudaVariable, CudaVariableNoGrad, myEmbedding, myLinear, myLSTM, biLSTM, PRU

class LM2(nn.Module):
    def __init__(self, args=None):
        super(LM2, self).__init__()
        self.dim_enc = args.dim_enc 
        self.dim_wemb = args.dim_wemb 
        self.max_length = args.max_length  

        self.src_emb = myEmbedding(args.data_words_n, args.dim_wemb) # src_emb
        self.rnn_enc = nn.LSTM(args.dim_wemb, args.dim_enc, batch_first=False, bidirectional=False) 
        self.dropout = nn.Dropout(p=args.dropout_p)

        self.readout = myLinear(args.dim_enc, args.dim_wemb)
        self.dec = myLinear(args.dim_wemb, args.data_words_n)
        self.dec.weight = self.src_emb.weight # weight tying.

    def init_hidden(self, Bn):
        hidden = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        cell = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        return hidden, cell
        
    def forward(self, data, mask=None): # if mask is None, assume generation 
        if mask is None:
            x_data = CudaVariable(torch.LongTensor(data))
            y_mask = None
        else:
            x_data = data[:-1]; x_data = CudaVariable(torch.LongTensor(x_data))
            y_data = data[1:]; y_data = CudaVariable(torch.LongTensor(y_data))
            y_mask = mask[1:]; y_mask = CudaVariable(torch.FloatTensor(y_mask))

        Tx, Bn = x_data.size() 
        x_emb = self.src_emb(x_data.view(Tx*Bn,1))
        x_emb = self.dropout(x_emb) 
 
        x_emb = x_emb.view(Tx,Bn,-1) 

        ht = CudaVariable(torch.zeros(1, Bn, self.dim_enc)) 
        ct = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        criterion = nn.NLLLoss(reduction='none')

        ht, ct = self.rnn_enc(x_emb); ht = self.dropout(ht) 
    
        output = self.readout(ht); output = self.dropout(output) # Tx Bn dim_wemb 
        logit = self.dec(output); logit = self.dropout(logit) # Tx Bn n_vocab

        probs = F.log_softmax(logit, dim=2) # Tx, Bn n_vocab 
        probs = probs.view(-1, probs.size(-1))

        if mask is not None:
            y_data = y_data.view(-1); y_mask = y_mask.view(-1) 
            loss = criterion(probs, y_data) # Bn n_vocab vs. Bn -> Bn
            loss = loss * y_mask
            n_words = torch.sum(y_mask).item()
            loss = torch.sum(loss) 
        else: # if mask None, assume generation
            loss = None
            n_words = None

        return loss, n_words, probs 

