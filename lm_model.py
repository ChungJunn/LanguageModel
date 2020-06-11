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
        x_data = data[:-1]
        y_data = data[1:]
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
        x_emb = self.src_emb(x_data.view(Tx*Bn,1)) # Tx Bn
        x_emb = x_emb.view(Tx,Bn,-1)

        ht = CudaVariable(torch.zeros(Bn, self.dim_enc))
        ct = CudaVariable(torch.zeros(Bn, self.dim_enc))
        loss = 0
        criterion = nn.NLLLoss(reduce=False)
        #criterion = nn.CrossEntropyLoss(reduce=False)
        for xi in range(Tx):
            ht, ct = self.rnn_enc.step(x_emb[xi,:,:], ht, ct, x_m=x_mask[xi])
            output = self.readout(ht)
            logit = self.dec(output)
            probs = F.log_softmax(logit, dim=1)
            #topv, yt = probs.topk(1)

            loss_t = criterion(probs, y_data[xi])
            if y_mask is not None:
                loss += torch.sum(loss_t*y_mask[xi])
            else:
                loss += torch.sum(loss_t)

        #if y_mask is not None:
        #    return loss/Bn
        #else:
        return loss/Bn

