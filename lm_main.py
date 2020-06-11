# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
import six; from six.moves import cPickle as pkl

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from lm_model import LM
from text_data import TextIterator

import re
from subprocess import Popen, PIPE

from mylib.utils import timeSince, ids2words, unbpe
import nmt_const as Const

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def train(model, optimizer, data, mask, args):

    loss = model(data, mask)

    model.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss.item()

def eval_model(model, data_file, args):
    torch.no_grad()
    valid_iter = TextIterator(data_file, args.data_dict,
                         batch_size=1, maxlen=args.max_length,
                         ahead=1, resume_num=0, mask_pos=False, const_id=Const)
    loss_total = 0.0
    for data, mask, cur_line, iloop in valid_iter:
        loss = model(data, mask)
        loss_total += loss.item()

    torch.set_grad_enabled(True)
    return loss_total / iloop

def train_model(args):
    # data loading
    train_iter = TextIterator(args.train_data_file, args.data_dict,
                         batch_size=args.batch_size, maxlen=args.max_length,
                         ahead=10, resume_num=0, mask_pos=False, const_id=Const)

    args.data_words_n = len(train_iter.data_dict2)

    start = time.time()
    loss_total = 0  # Reset every args.print_every

    # model 
    model = LM(args=args).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    file_name = args.save_dir + '/' + args.model_file + '.pth'
    with open(file_name+'.loss', 'w') as bfp:
        bfp.write('0 val: 0 test: 0\n')

    for data, mask, cur_line, iloop in train_iter:

        loss = train(model, optimizer, data, mask, args)
        loss_total += loss

        if iloop % args.print_every == 0:
            loss_avg = loss_total/args.print_every
            loss_total = 0
            print('%s: %d iters - %.4f %s' % (args.model_file, iloop, loss_avg, timeSince(start)))

        VAL_START = 1000
        if iloop >= VAL_START and iloop % args.valid_every == 0:
            print ('saving the model to '+file_name)
            #torch.save(model, file_name)

            val_loss = eval_model(model, args.valid_data_file, args)
            #print('val -> test')
            test_loss = eval_model(model, args.test_data_file, args)

            if iloop > VAL_START and os.path.exists(file_name+'.loss'):
                with open(file_name+'.loss', 'r') as bfp:
                    lines = bfp.readlines()
                    prev_loss = [float(bs.split()[2]) for bs in lines]

                if val_loss <= np.min(prev_loss):
                    #torch.save(model, file_name +'.best.pth')
                    print('the best model is saved to '+file_name+'.best.pth')

            with open(file_name+'.loss', 'a') as bfp:
                bfp.write(str(iloop) + ' val: ' + str(val_loss) + ' test: ' + str(test_loss) + '\n')

            print ('valid loss', val_loss, 'test_loss', test_loss)

