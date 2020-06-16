###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--dict_path', type=str, default='/home/chl/data/wikitext-2-raw/wiki.train.raw.voc.pkl',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./result/LAN-99.pth.best.pth',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")


import nmt_const as Const
from text_data import read_dict

dict = read_dict('/home/chl/data/wikitext-2-raw/wiki.train.raw.voc.pkl', Const)
idx2word = {}
idx = 0
for kk, vv in dict.items():
    idx2word[vv] = kk
n_tokens = len(dict) #TODO

'''
# testing model
args.dim_enc=100
args.dim_wemb=100
args.max_length=100
args.data_words_n=n_tokens
args.dropout_p=0.5

from lm_model import LM2
model = LM2(args).to(device) 
model.eval()
'''

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

hidden = model.init_hidden(1)
input = np.random.randint(n_tokens, size=(1, 1)).astype(np.int32)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            _, probs = model(input)
            word_weights = probs.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill(word_idx.item())

            word = idx2word[word_idx.item()] #TODO

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
