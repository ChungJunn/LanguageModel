# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import time
import math

import torch 
import torch.nn as nn
from torch import optim

from lm_model import LM2
from text_data import TextIterator

from mylib.utils import timeSince, ids2words, unbpe
import nmt_const as Const
import sys

use_cuda = torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

def train(model, optimizer, data, mask, args):

    loss, n_words, _ = model(data, mask) # a, b not used

    model.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss.item(), n_words

def eval_model(model, data_file, args):
    model.eval() 
    torch.no_grad()
    valid_iter = TextIterator(data_file, args.data_dict,
                         batch_size=1, maxlen=args.max_length,
                         ahead=1, resume_num=0, mask_pos=False, const_id=Const)
    loss_total = 0.0
    n_total = 0.0
    for data, mask, cur_line, iloop in valid_iter:
        loss, n_words, _ = model(data, mask)
        loss_total += loss.item()
        n_total += n_words

    torch.set_grad_enabled(True)
    return loss_total / n_total 

def train_model(args, neptune):
    # data loading
    train_iter = TextIterator(args.train_data_file, args.data_dict,
                         batch_size=args.batch_size, maxlen=args.max_length,
                         ahead=10, resume_num=0, mask_pos=False, const_id=Const)
 
    args.data_words_n = len(train_iter.data_dict2)
    start = time.time()
    loss_total = 0  # Reset every args.print_every
    words_total = 0

    # model 
    model = LM2(args=args).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.0, 0.999))

    file_name = args.save_dir + '/' + args.exp_id + '.pth'
    bad_counter=0
    best_loss=None
    
    try: 
        for data, mask, cur_line, iloop in train_iter:
            loss, n_words = train(model, optimizer, data, mask, args)
            loss_total += loss
            words_total += n_words 

            if iloop % args.print_every == 0:
                loss_avg = loss_total / words_total
                loss_total = 0; words_total = 0
                print('%s: %d iters - %.4f %s' % (args.name + args.tag, iloop, loss_avg, timeSince(start)))
                neptune.log_metric('train loss', loss_avg)
                neptune.log_metric('train ppl', math.exp(loss_avg)) 

            if iloop >= args.val_start and iloop % args.valid_every == 0:
                val_loss = eval_model(model, args.valid_data_file, args)
                model.train()

                if iloop > args.val_start: 
                    if best_loss is None or val_loss <= best_loss: 
                        bad_counter = 0
                        torch.save(model, file_name +'.best.pth')
                        best_loss = val_loss
                    else:
                        bad_counter += 1
                        print('bad_counter:', bad_counter)

                    if bad_counter >= args.patience:
                        flag = False
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= args.lr_decay
                            
                            if not flag:
                               print('lr decayed to {:.4f}'.format(param_group['lr'])) 
                               flag = True

                        bad_counter = 0

                print ('valid loss', val_loss)
                neptune.log_metric('valid loss', val_loss)
                neptune.log_metric('valid ppl', math.exp(val_loss)) 
    
    except KeyboardInterrupt:
        test_loss = eval_model(model, args.test_data_file, args)
        print ('test loss {:.4f} ppl: {:.4f}'.format(test_loss, math.exp(test_loss)))
        neptune.log_metric('test loss', test_loss)
        neptune.log_metric('test ppl', math.exp(test_loss)) 
