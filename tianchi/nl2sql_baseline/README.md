git checkout  -b python3  origin/python3

## More Intro https://new.qq.com/rain/a/20190602A089LI

## Introduction

This baseline method is developed and refined based on <a href="https://github.com/xiaojunxu/SQLNet">code</a> of <a href="https://arxiv.org/abs/1711.04436">SQLNet</a>, which is a baseline model in <a href="https://github.com/salesforce/WikiSQL">WikiSQL</a>.

The model decouples the task of generating a whole SQL into several sub-tasks, including select-number, select-column, select-aggregation, condition-number, condition-column and so on.

Simple model structure shows here, implementation details could refer to the origin <a href="https://arxiv.org/abs/1711.04436">paper</a>.

<div align="middle"><img src="https://github.com/ZhuiyiTechnology/nl2sql_baseline/blob/master/img/detailed_structure.png"width="80%" ></div>

The difference between SQLNet and this baseline model is, Select-Number and Where-Relationship sub-tasks are added to adapt this Chinese NL2SQL dataset better.

## Dependencies

 - Python 2.7
 - torch 1.0.1
 - tqdm

## Start to train

Firstly, download the provided datasets at ~/data_nl2sql/, which includes train.json, train.tables.json, dev.json, dev.tables.json and char_embedding.
```
├── data
│ ├── train
│ │ ├── train.json
│ │ ├── train.tables.json
│ ├── dev
│ │ ├── dev.json
│ │ ├── dev.tables.json
├── char_embedding
```

```
mkdir ~/nl2sql
cd ~/nl2sql/
git clone https://github.com/ZhuiyiTechnology/nl2sql_baseline.git

cp ~/data_nl2sql/* ~/nl2sql/nl2sql_baseline/data
cd ~/nl2sql/nl2sql_baseline/

sh ./start_train.py 0 128
```
while the first parameter 0 means gpu number, the second parameter means batch size.

## Start to evaluate

To evaluate on dev.json or test.json, make sure trained model is ready, then run
```
cd ~/nl2sql/nl2sql_baseline/
sh ./start_test.py 0 pred_example
```
while the first parameter 0 means gpu number, the second parameter means the output path of prediction.

## Experiment result

We have run experiments several times, achiving avegrage 27.5% logic form accuracy on the dev dataset.


## Experiment analysis

We found the main challenges of this datasets containing poor condition value prediction, select column and condition column not mentioned in NL question, inconsistent condition relationship representation between NL question and SQL, etc. All these challenges could not be solved by existing baseline and SOTA models.

Correspondingly, this baseline model achieves only 77% accuracy on condition column and 62% accuracy on condition value respectively even on the training set, and the overall logic form is only around 50% as well, indicating these problems are challenging for contestants to solve.

<div align="middle"><img src="https://github.com/ZhuiyiTechnology/nl2sql_baseline/blob/master/img/trainset_behavior.png"width="80%" ></div>

## Related resources:
https://github.com/salesforce/WikiSQL

https://yale-lily.github.io/spider

<a href="https://arxiv.org/pdf/1804.08338.pdf">Semantic Parsing with Syntax- and Table-Aware SQL Generation</a>










###  Prepare envirment 
1. Select and download the corresponding version  anaconda from [Anaconda](https://www.anaconda.com/)
2. Install anaconda
``` shell 
bash Anaconda*.sh  
```
3. Creating a Virtual Environment
``` shell
conda create --name python27 python=2.7
conda activate python27
pip install torchvison 
pip install torch 
pip install records 
pip install babel 
pip install tqdm 
``` 
4. Get the data  
[赛题与数据](https://tianchi.aliyun.com/competition/entrance/231716/score) 

5. Runing code 
```python 
python train.py --ca --gpu   # must used --gpu param, otherwise it will throw some unexpected  errors 
``` 

6. Data explore   
本次赛题将提供4万条有标签数据作为训练集，1万条无标签数据作为测试集。其中，5千条测试集数据作为初赛测试集，对选手可见；5千条作为复赛测试集，对选手不可见。

提供的数据集主要由3个文件组成，以训练集为例，包括train.json、train.tables.json及train.db。

train.json文件中，每一行为一条数据样本。数据样例及字段说明例如下：
``` json 
{
     "table_id": "a1b2c3d4", # 相应表格的id
     "question": "世茂茂悦府新盘容积率大于1，请问它的套均面积是多少？", # 自然语言问句
     "sql":{ # 真实SQL
        "sel": [7], # SQL选择的列 
        "agg": [0], # 选择的列相应的聚合函数, '0'代表无
        "cond_conn_op": 0, # 条件之间的关系
        "conds": [
            [1,2,"世茂茂悦府"], # 条件列, 条件类型, 条件值，col_1 == "世茂茂悦府"
            [6,0,1]
        ]
    }
}
``` 
其中，SQL的表达字典说明如下：
``` json 
op_sql_dict = {0:">", 1:"<", 2:"==", 3:"!="}
agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
conn_sql_dict = {0:"and",    1:"or",   -1:""}
``` 
train.tables.json 文件中，每一行为一张表格数据。数据样例及字段说明例如下：
``` json 
{
    "id":"a1b2c3d4", # 表格id
    "name":"Table_a1b2c3d4", # 表格名称
    "title":"表1：2019年新开工预测 ", # 表格标题
    "header":[ # 表格所包含的列名
        "300城市土地出让",
        "规划建筑面积(万㎡)",
        ……
    ],
    "types":[ # 表格列所相应的类型
        "text",
        "real",
        ……
    ],
    "rows":[ # 表格每一行所存储的值
        [
            "2009年7月-2010年6月",
            168212.4,
            ……
        ]
    ]
}
``` 
tables.db为sqlite格式的数据库形式的表格文件。各个表的表名为tables.json中相应表格的name字段。为避免部分列名中的特殊符号导致无法存入数据库文件，表格中的列名为经过归一化的字段，col_1, col_2, …, col_n。db文件将后续更新。

另外，也提供用于baseline方案的字向量文件char_embedding，每一行的内容为字符及其300维的向量表达，以空格分隔。


7. Execute log 
```
Loading dataset
Loading data from data/dev.json
Loading data from data/dev.tables.json
Loading data from data/train.json
Loading data from data/train.tables.json
Loading word embedding from data/char_embedding
Using fixed embedding
Using column attention on select number predicting
Using column attention on selection predicting
Using column attention on aggregator predicting
Using column attention on where predicting
Using column attention on where relation predicting
####################  Star to Train  ####################
Epoch 1
100%|██████████| 2306/2306 [05:26<00:00,  7.02it/s]
100%|██████████| 275/275 [00:34<00:00,  7.94it/s]
Sel-Num: 0.881, Sel-Col: 0.151, Sel-Agg: 0.749, W-Num: 0.684, W-Col: 0.402, W-Op: 0.655, W-Val: 0.325, W-Rel: 0.517
Train loss = 5.331
Dev Logic Form: 0.000
Best Logic Form: 0.000 at epoch 1
Epoch 2
100%|██████████| 2306/2306 [05:42<00:00,  7.05it/s]
100%|██████████| 275/275 [00:33<00:00,  8.78it/s]
Sel-Num: 0.881, Sel-Col: 0.151, Sel-Agg: 0.749, W-Num: 0.681, W-Col: 0.398, W-Op: 0.654, W-Val: 0.326, W-Rel: 0.522
Train loss = 4.852
Dev Logic Form: 0.000
Best Logic Form: 0.000 at epoch 1
Epoch 3
100%|██████████| 2306/2306 [06:00<00:00,  6.55it/s]
 81%|████████  | 223/275 [00:27<00:05,  9.03it/s]

``` 
