#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/5/13 10:28
#@Author: yangjian
#@File  : model.py

import numpy
import os
import torch
from flyai.model.base import Base

__import__('net', fromlist=["Net"])

MODEL_NAME = "model.pkl"
from path import MODEL_PATH

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        network = torch.load(os.path.join(MODEL_PATH, MODEL_NAME))
        x_data = self.data.predict_data(**data)
        x_1 = torch.from_numpy(x_data[0])
        x_2 = torch.from_numpy(x_data[1])
        x_1 = x_1.float().to(device)
        x_2 = x_2.float().to(device)
        outputs = network(x_1, x_2)
        _, prediction = torch.max(outputs.data, 1)
        prediction = prediction.cpu()
        prediction = prediction.numpy()
        return prediction

    def predict_all(self, datas):
        network = torch.load(os.path.join(MODEL_PATH, MODEL_NAME))
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            x_1 = torch.from_numpy(x_data[0])
            x_2 = torch.from_numpy(x_data[1])
            x_1 = x_1.float().to(device)
            x_2 = x_2.float().to(device)
            outputs = network(x_1, x_2)
            _, prediction = torch.max(outputs.data, 1)
            prediction = prediction.cpu()
            prediction = prediction.numpy()
            labels.append(prediction)
        return labels

    def batch_iter(self, x1, x2, y, batch_size=16):
        """生成批次数据"""
        data_len = len(x1)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x1_shuffle = x1[indices]
        x2_shuffle = x2[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x1_shuffle[start_id:end_id], x2_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))