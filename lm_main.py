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

def train_model(args, neptune):
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

    file_name = args.save_dir + '/' + args.exp_id + '.pth'
    bad_counter=0
    best_loss=None
 
    for data, mask, cur_line, iloop in train_iter:
        loss = train(model, optimizer, data, mask, args)
        loss_total += loss

        if iloop % args.print_every == 0:
            loss_avg = loss_total/args.print_every
            loss_total = 0
            print('%s: %d iters - %.4f %s' % (args.name + args.tag, iloop, loss_avg, timeSince(start)))
            neptune.log_metric('train loss', loss_avg)

        if iloop >= args.val_start and iloop % args.valid_every == 0:
            val_loss = eval_model(model, args.valid_data_file, args)
            test_loss = eval_model(model, args.test_data_file, args)

            if iloop > args.val_start: 
                if best_loss is None or val_loss <= best_loss: 
                    bad_counter = 0
                    torch.save(model, file_name +'.best.pth')
                    best_loss = val_loss
                else:
                    bad_counter += 1
                    print('bad_counter:', bad_counter)

                if bad_counter > args.patience:
                    print('Early Stopping')
                    break 

            with open(file_name+'.loss', 'a') as bfp:
                bfp.write(str(iloop) + ' val: ' + str(val_loss) + ' test: ' + str(test_loss) + '\n')
            print ('valid loss', val_loss, 'test_loss', test_loss)
            neptune.log_metric('valid loss', val_loss)

