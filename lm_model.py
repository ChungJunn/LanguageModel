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

class LM(nn.Module):
    def __init__(self, args=None):
        super(LM, self).__init__()
        self.dim_enc = args.dim_enc # dimension encoding
        self.dim_wemb = args.dim_wemb # dimension of word embedding
        self.max_length = args.max_length # 
        self.rnn_name = args.rnn_name # 
        self.loss_reduction = args.loss_reduction

        self.src_emb = myEmbedding(args.data_words_n, args.dim_wemb) # src_emb

        if self.rnn_name == 'lstm':
            self.rnn_enc = nn.LSTM(args.dim_wemb, args.dim_enc, batch_first=False, bidirectional=False)
        elif self.rnn_name == 'mylstm':
            self.rnn_enc = myLSTM(args.dim_wemb, args.dim_enc, batch_first=False, direction='f')
        elif self.rnn_name == 'pru':
            if args.rnn_ff==1:
                self.rnn_enc = PRU(args.dim_wemb, args.dim_enc, batch_first=False, direction='f', plus=True)
            else:
                self.rnn_enc = PRU(args.dim_wemb, args.dim_enc, batch_first=False, direction='f', plus=False)

        self.readout = myLinear(args.dim_enc, args.dim_wemb)
        self.dec = myLinear(args.dim_wemb, args.data_words_n)
        self.dec.weight = self.src_emb.weight # weight tying.
        # how does this work?

    def init_hidden(self, Bn):
        hidden = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        cell = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        return hidden, cell
        
    def forward(self, data, mask=None):
        x_data = data[:-1] # create target  
        y_data = data[1:] # create labels
        x_data = CudaVariable(torch.LongTensor(x_data))
        y_data = CudaVariable(torch.LongTensor(y_data))

        if mask is None:
            x_mask = None
            y_mask = None
        else:
            x_mask = mask[1:]
            y_mask = mask[1:]
            x_mask = CudaVariable(torch.FloatTensor(x_mask))
            y_mask = CudaVariable(torch.FloatTensor(y_mask))

        Tx, Bn = x_data.size() 
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # Tx Bn #TODO: study about view function
        x_emb = x_emb.view(Tx,Bn,-1) # forward through embedding and restore
        # TODO embeding output Tx Bn dim_wemb

        ht = CudaVariable(torch.zeros(Bn, self.dim_enc)) # initialize the hidden states
        ct = CudaVariable(torch.zeros(Bn, self.dim_enc))
        loss = 0
        criterion = nn.NLLLoss(reduction='none')
        #criterion = nn.CrossEntropyLoss(reduce=False)
        denum=0

        for xi in range(Tx):
            ht, ct = self.rnn_enc.step(x_emb[xi,:,:], ht, ct, x_m=x_mask[xi]) # Bn dim_enc  
            output = self.readout(ht) # Bn dim_wemb - check 
            logit = self.dec(output) # Bn n_vocab - check
            probs = F.log_softmax(logit, dim=1) # Bn n_vocab - check
            #topv, yt = probs.topk(1)

            loss_t = criterion(probs, y_data[xi]) # Bn n_vocab vs. Bn -> Bn
            
            if self.loss_reduction == 'org': # Original: prof's way (average within time-step first then average across batches)
                if y_mask is not None:
                    loss += torch.mean(loss_t*y_mask[xi]) # Bn -> 1 
                else:
                    loss += torch.mean(loss_t)

            if self.loss_reduction == 'con': # Conventional: average across time-step and batches
                if y_mask is not None:
                    loss += torch.sum(loss_t*y_mask[xi]) # Bn -> 1 
                else:
                    loss += torch.sum(loss_t)
       
        if self.loss_reduction == 'org': denum = Bn
        if self.loss_reduction == 'con': denum = torch.sum(y_mask).item()
        
        return loss / denum

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
        
    def forward(self, data, mask=None):
        x_data = data[:-1]; x_data = CudaVariable(torch.LongTensor(x_data))
        y_data = data[1:]; y_data = CudaVariable(torch.LongTensor(y_data))
  
        if mask is None:
            y_mask = None
        else:
            y_mask = mask[1:]; y_mask = CudaVariable(torch.FloatTensor(y_mask))

        Tx, Bn = x_data.size() 
        x_emb = self.src_emb(x_data.view(Tx*Bn,1))
        x_emb = self.dropout(x_emb) 
 
        x_emb = x_emb.view(Tx,Bn,-1) 

        ht = CudaVariable(torch.zeros(1, Bn, self.dim_enc)) 
        ct = CudaVariable(torch.zeros(1, Bn, self.dim_enc))
        criterion = nn.NLLLoss(reduction='none')

        ht, ct = self.rnn_enc(x_emb); ht = self.dropout(ht) 
    
        output = self.readout(ht); output = self.dropout(output) # Bn dim_wemb 
        logit = self.dec(output); logit = self.dropout(logit) # Bn n_vocab

        probs = F.log_softmax(logit, dim=2) # Bn n_vocab 
        probs = probs.view(-1, probs.size(-1))
        y_data = y_data.view(-1); y_mask = y_mask.view(-1)
        
        loss = criterion(probs, y_data) # Bn n_vocab vs. Bn -> Bn
        loss = loss * y_mask
        loss = torch.sum(loss) / torch.sum(y_mask) 
        return loss 
