# -*- coding: utf-8 -*
'''
实现模型的调用
'''
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict(source="新华社 北京 9 月 11 日电 　 第二十二届 国际 检察官 联合会 年会 暨 会员 代表大会 11 日 上午 在 北京 开幕 。 国家 主席 习近平 发来 贺信 ， 对 会议 召开 表示祝贺 。")
print(p)
