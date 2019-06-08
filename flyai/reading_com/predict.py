#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/5/13 10:28
#@Author: yangjian
#@File  : predict.py

from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(question='增城公交路线到达增城区新塘镇的有哪些？',
                  answer='新塘镇内公交车（主要行走于增城区新塘镇内，15条线路）；各镇间公交车（穿梭于各镇街，15条线路），总共61条线路。')
print(p)