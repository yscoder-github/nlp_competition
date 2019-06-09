# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(sentence="It is obvious that the more you eat, the less you want.")
print(p)
