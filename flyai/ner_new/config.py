# -*- coding: utf-8 -*-
import os
import json
import path


src_vocab_file = os.path.join(path.DATA_PATH ,'words.dict')
word_embedding_file = os.path.join(path.DATA_PATH ,'embedding.json')
embeddings_size = 200
max_sequence = 100
dropout = 0.6
leanrate = 0.001

with open(os.path.join(path.DATA_PATH, 'words.dict'), 'r') as vocab_file:
    vocab_size = len(json.load(vocab_file))

src_unknown_id = vocab_size
src_padding = vocab_size + 1
label_dic = ['B-LAW', 'B-ROLE', 'B-TIME', 'I-LOC','I-LAW','B-PER','I-PER','B-ORG','I-ROLE','I-CRIME','B-CRIME','I-ORG','B-LOC','I-TIME','O','padding']
label_len = len(label_dic)