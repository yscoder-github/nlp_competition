####[NLP]基于TextCNN的文本情感分类

### 1. 模型原理
##### 1.1 论文
Yoon Kim在论文[(2014 EMNLP) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)提出TextCNN。

将卷积神经网络CNN应用到文本分类任务，利用多个不同大小的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。

##### 1.2 网络结构

![avatar](./pic/textcnn_model_arch.png)

TextCNN的详细过程原理图如下：

![](./pic/textcnn_brief.png)


TextCNN主要包含如下几层:

1. Embedding：第一层是图中最左边的7*5的句子矩阵，每行是词向量，维度为5，这个可以类比为图像中的原始像素点。
2. Convolution：经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
  每个kernel都有其"感受野“，在上图中三个kernel的**一维感受野**分别为2,3,4
3. MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
4. FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。
  

其他相关概念介绍:
通道（Channels）：

图像中可以利用 (R, G, B) 作为不同channel；
文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。
 

一维卷积（conv-1d）：

图像是二维数据；
文本是一维数据，因此在TextCNN卷积用的是一维卷积（在word-level上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要通过设计不同 kernel_size 的 filter 获取不同宽度的视野。
 

Pooling层：

利用CNN解决文本分类问题的文章还是很多的，比如这篇 A Convolutional Neural Network for Modelling Sentences 最有意思的输入是在 pooling 改成 (dynamic) k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息。

比如在情感分析场景，举个例子：

“我觉得这个地方景色还不错，但是人也实在太多了”
 虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用 k-max pooling能够很好捕捉这类信息。

 
2. 实现
基于Tensorflow深度学习框架的实现代码如下()
```python 



```

特征：这里用的是词向量表示方式

数据量较大：可以直接随机初始化embeddings，然后基于语料通过训练模型网络来对embeddings进行更新和学习。
数据量较小：可以利用外部语料来预训练(pre-train)词向量，然后输入到Embedding层，用预训练的词向量矩阵初始化embeddings。（通过设置weights=[embedding_matrix]）。
静态(static)方式：训练过程中不再更新embeddings。实质上属于迁移学习，特别是在目标领域数据量比较小的情况下，采用静态的词向量效果也不错。（通过设置trainable=False）
非静态(non-static)方式：在训练过程中对embeddings进行更新和微调(fine tune)，能加速收敛。（通过设置trainable=True）



plot_model()画出的TextCNN模型结构图如下：


