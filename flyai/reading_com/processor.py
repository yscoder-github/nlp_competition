#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/5/13 10:28
#@Author: yangjian
#@File  : processor.py

import sys
import os
import re
import jieba
import codecs
import json
import numpy as np

from flyai.processor.base import Base
from path import DATA_PATH  # 导入输入数据的地址


sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        embedding_path = os.path.join(DATA_PATH, 'embedding.json')
        with open(embedding_path, encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        self.max_sts_len = 10
        self.embedding_len = 100

    def input_x(self, question, answer):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        question = str(question)
        word_list1 = []
        question = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", question)
        question = jieba.cut(question, cut_all=False)
        for word in question:
            embedding_vector = self.vocab.get(word)
            if embedding_vector is not None:
                if len(embedding_vector) == self.embedding_len:
                    # 给出现在编码词典中的词汇编码
                    embedding_vector = list(map(lambda x: float(x),
                                                embedding_vector))  ## convert element type from str to float in the list
                    word_list1.append(embedding_vector)
        if len(word_list1) >= self.max_sts_len:
            word_list1 = word_list1[:self.max_sts_len]
        else:
            for i in range(len(word_list1), self.max_sts_len):
                word_list1.append([0 for j in range(self.embedding_len)])  ## 词向量维度为200
        word_list1 = np.stack(word_list1)

        answer = str(answer)
        word_list2 = []
        answer = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", answer)
        answer = jieba.cut(answer, cut_all=False)
        for word in answer:
            embedding_vector = self.vocab.get(word)
            if embedding_vector is not None:
                if len(embedding_vector) == self.embedding_len:
                    # 给出现在编码词典中的词汇编码
                    embedding_vector = list(map(lambda x: float(x),
                                                embedding_vector))  ## convert element type from str to float in the list
                    word_list2.append(embedding_vector)
        if len(word_list2) >= self.max_sts_len:
            word_list2 = word_list2[:self.max_sts_len]
        else:
            for i in range(len(word_list2), self.max_sts_len):
                word_list2.append([0 for j in range(self.embedding_len)])  ## 词向量维度为200
        word_list2 = np.stack(word_list2)
        return word_list1, word_list2

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels = np.array(data)
        labels = labels.astype(np.float32)
        out_y = labels
        return out_y
