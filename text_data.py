# -*- coding: utf-8 -*-

import io
import six; from six.moves import cPickle as pkl
import gzip
import numpy as np
from mylib.utils import equizip

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return io.open(filename, mode, encoding="utf-8")

def read_dict(dic_file, const_id=None):
    with open(dic_file, 'rb') as f:
        src_dict = pkl.load(f, encoding="utf-8")
    src_dict2 = dict()
    for kk, vv in src_dict.items():
        src_dict2[kk] = vv+2 # in the dict file, <s>/</s>=0, <unk>=1
    if const_id is None:
        src_dict2['PAD'] = 0
        src_dict2['<s>'] = 1
    else:
        src_dict2['PAD'] = const_id.PAD
        src_dict2['<s>'] = const_id.BOS
    return src_dict2

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, data_dict, 
                 unk_id=2, batch_size=128, maxlen=50, 
                 ahead=1, resume_num=0, just_epoch=0, mask_pos=False, const_id=None):
        self.source_name = source
        self.unk_id = unk_id
        self.const_id = const_id

        self.source = fopen(source, 'r')
        self.data_dict2 = read_dict(data_dict, const_id=const_id)
        self.mask_pos = mask_pos # for Att is All You Need.

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.end_of_data = False

        self.just_epoch = just_epoch
        self.x_buf =[]
        self.buf_remain = 0
        self.cur_line_num=0
        self.ahead=ahead
        self.iters = 0

        if resume_num > 0:
            self.cur_line_num=resume_num
            for i in range(resume_num):
                ss = self.source.readline()

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.cur_line_num=0
        self.iters = self.iters + 1

    def __next__(self):

        if self.buf_remain == 0:
            self.x_buf = []
            i = 0
            while True:
                ss = self.source.readline()
                if ss == "":
                    self.reset()
                    if self.ahead == 1 or self.just_epoch:
                        raise StopIteration # validation
                    ss = self.source.readline()

                ss = ss.strip().split()
                if len(ss) <= 1:
                    continue
                ss = [self.data_dict2.get(key, self.unk_id) for key in ss] # 0 BOS, 1 EOS, 2 UNK
                self.cur_line_num = self.cur_line_num + 1

                if len(ss) > self.maxlen:
                    continue

                self.x_buf.append(ss) # given sentence is appended to the x_buf

                if len(self.x_buf) >= self.batch_size*self.ahead:
                    break

            self.buf_remain = self.ahead # TODO I don't understand

            len_xs = [(len(x), x) for x in self.x_buf]
            sorted_len_xs = sorted(len_xs, key=lambda xs: xs[0])
            self.x_buf = [xs[1] for xs in sorted_len_xs]

        # with self.buf_remain as index
        br = self.ahead-self.buf_remain
        bs = self.batch_size

        source = self.x_buf[br*bs:(br+1)*bs]

        self.buf_remain = self.buf_remain - 1

        x_data, x_mask = self.prepare_text(source)
        self.iters = self.iters + 1
        return x_data, x_mask, self.cur_line_num, self.iters

    # batch preparation, returns padded batch and mask
    def prepare_text(self, seqs_x):
        # x: a list of sentences
        lengths_x = [len(s) for s in seqs_x] 
        n_samples = len(seqs_x)

        maxlen_x = np.max(lengths_x) + 2 # +2 for BOS and EOS

        x_data = np.ones((maxlen_x, n_samples)).astype('int64')*self.const_id.PAD # EOS_token = 1
        x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
        #import pdb;pdb.set_trace()
        for idx, s_x in enumerate(seqs_x): # sentence is given as list of strings
            x_data[1:lengths_x[idx]+1, idx] = s_x
            x_data[0, idx] = self.const_id.BOS
            x_data[lengths_x[idx]+1, idx] = self.const_id.EOS
            if self.mask_pos:
                x_mask[:lengths_x[idx]+2, idx] = 1.+np.arange(lengths_x[idx]+2) # +2 for BOS and EOS
            else:
                x_mask[:lengths_x[idx]+2, idx] = 1. # +2 for BOS and EOS

        return x_data[1:,:], x_mask[1:,:]


if __name__ == "__main__":
    import nmt_const as Const
    base_dir = '/home/chl/data/wikitext-2-raw'
    src_file = base_dir + '/wiki.test.raw'
    src_dict = base_dir + '/wiki.train.raw.voc.pkl'

    train_iter = TextIterator(src_file, src_dict,
                         batch_size=3, maxlen=300, 
                         ahead=5, resume_num=0, mask_pos=False, const_id=Const)
    
    idx = 0
    for x, xm, tmp1, ii in train_iter:
        import pdb; pdb.set_trace()
        print ('==============================')
        print (len(x))
        print (x)
        print (xm)
        print (tmp1)
        idx = idx + 1
        if idx > 10: 
            break
        
