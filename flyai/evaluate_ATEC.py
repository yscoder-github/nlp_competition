import sys

import json
import random
from flyai.dataset import Dataset

from model import Model

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor(sys.argv[1])

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(y_test)

model = Model(dataset)
labels = model.predict_all(x_test)
eval = 0

if len(y_test) != len(labels):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    for index in range(len(labels)):
        label = labels[index]
        test = y_test[index]
        if 'label' in str(label):
            label = label['label']
        if label == test['label']:
            eval = eval + 1
    eval = eval / len(labels)

result = dict()
result['score'] = round(eval * 100, 2)
result['label'] = "分数为准确率"
result['info'] = ""
print(json.dumps(result))
