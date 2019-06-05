# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(texta="还款日过了，为什么花呗不能用",textb='花呗逾期怎么不能还款')
print(p)
