#! -*- coding:utf-8 -*-


import json
from tqdm import tqdm
import codecs
import os 

all_50_schemas = set()

base_path = '/home/yinshuai/文档/baidu_info_extract'

with open(os.path.join(base_path, 'all_50_schemas')) as f:
    for l in tqdm(f):
        a = json.loads(l)
        all_50_schemas.add(a['predicate'])


id2predicate = {i:j for i,j in enumerate(all_50_schemas)}
predicate2id = {j:i for i,j in id2predicate.items()}


with codecs.open(os.path.join(base_path, 'all_50_schemas_me.json'), 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


chars = {}
min_count = 2


train_data = []


with open(os.path.join(base_path, 'train_data.json')) as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['spo_list']:
            continue
        train_data.append(
            {
                'text': a['text'],
                'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1


with codecs.open(os.path.join(base_path, 'train_data_me.json'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


dev_data = []


with open(os.path.join(base_path, 'dev_data.json')) as f:
    for l in tqdm(f):
        a = json.loads(l)
        dev_data.append(
            {
                'text': a['text'],
                'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1


with codecs.open(os.path.join(base_path, 'dev_data_me.json'), 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)


with codecs.open(os.path.join(base_path, 'all_chars_me.json'), 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
