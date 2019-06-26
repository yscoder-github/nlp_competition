# -*- coding: UTF-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK, our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print "Using fixed embedding"


    def gen_x_batch(self, q, col):
        '''
        get input question batch 
        input: 
              q: char-based questions 
              col: header of each  table  [batch_size, headers_of_current_table]
        return:
               val_inp_var: 
               val_len: each question length in one batch 
        '''
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            if self.trainable:
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            else: # q_val: char-embedding of question 
                q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)
            if self.our_model:
                if self.trainable:
                    val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                else:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else: # default not go here
                one_col_all = [x for toks in one_col for x in toks + [',']]
                if self.trainable:
                    col_val = map(lambda x:self.w2i.get(x, 0), one_col_all)
                    val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + q_val+ [0])
                else:
                    col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_col_all)
                    val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.N_word, dtype=np.float32)] + q_val+ [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B): # list to numpy?? why use this complex method to trans? 
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        '''
        input: 
                cols: columns in table(get from header)
                    eg:[['索', '书', '号'], ['书', '名'], ['编', '著', '者'], ['出', '版', '社'], ['出', '版', '时', '间'], ['册', '数']]
                    [batch_size(16), all_columns_in_current_train_table]
        output: 
               name_inp_var: word embedding of columns [batch_size(not question batch_size,but column), max_len(column),hidden_size]
               name_len: column length, eg: name_len = length('容积率')=3 
               col_len:  count of columns in each table   [batch_size(not 16), ]
        '''
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        '''
        input: 
            str_list: all columns in one batch, not one table 
                eg:  [[u'图', u'书', u'编', u'号'], [u'S', u'S', u'号'], [u'书', u'名'], [u'作', u'者'], [u'出', u'版', u'社'], [u'出', u'版', u'日', u'期'], [u'i', u's', u'b', u'n'], [u'内', u'容', u'简', u'介'], [u'分', u'类', u'名', u'称'], [u'商', u'户', u'名', u'称'], [u'门', u'店', u'名', u'称'], [u'商', u'户', u'营', u'业', u'地', u'址'], [u'区', u'号'], [u'商', u'户', u'咨', u'询', u'电', u'话'], ...]
        output:  val_inp_var: each  column  embedding    [batch_size(not 16), max_len(n_step), hidden_size]
                 val_len: each column length  [batch_size(not 16)]
            
        '''
        # at here B<> 16 and it equals to column counts in one batch 
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable: # default not go here 
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len) # max column length  

        if self.trainable: # not go here 
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i, t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else: # go here 
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len
