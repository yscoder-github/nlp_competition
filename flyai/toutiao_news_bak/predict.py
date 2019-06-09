# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(news="赵丽颖很久没有登上微博热搜了，但你们别急，她只是在憋大招而已”！")
print(p)
