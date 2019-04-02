"""
RNN(Recurrent Neural Network)
- Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)

many-to-one model !!



The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset.
The train/test split has been preserved for the purposes of benchmarking,
but the sentences have been shuffled from their original order.
Each Sentence has been parsed into many phrases by the Stanford parser.
Each phrase has a PhraseId. Each sentence has a SentenceId.
 Phrases that are repeated (such as short/common words) are only included once in the data.

train.tsv contains the phrases and their associated sentiment labels.
 We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:

0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

"""

import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from nltk.corpus import stopwords

__author__ = 'yscoder@foxmail.com'
tf.reset_default_graph()

# Data PreProcess
df_train = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/train.tsv', sep='\t')
df_test = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/test.tsv', sep='\t')


print(df_train)
phrase = df_train['Phrase'].tolist()
max_len = 0
for p in phrase:
    if len(p) > max_len:
        max_len = len(p)

print(max_len)



# sys.exit(0)


labels = df_train['Sentiment'].tolist()


word_list = " ".join(phrase).split()
word_list = list(set(word_list))
stop_word = set(stopwords.words('english'))

word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

"""
  code by  yinshuai
  Semantic analyse -- 情感分析,分析语句的情感, CV中"感受野"的概念要懂得迁移到NLP上
"""
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

# Text-CNN Parameter
embedding_size = 5  # n-gram
sequence_length = 284  # 每个句子的长度都是3
num_classes = 5  # 0 or 1 两种情感
filter_sizes = [2, 2, 2]  # n-gram window
num_filters = 3

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
# labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.


sentences = phrase
print(sentences)
print(type(sentences))

# labels = labels



word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}  # 词汇表
vocab_size = len(word_dict)  # 词汇表



def build_dataset(words, n_words, atleast=1):
    """
    构建数据集
    :param words:
    :param n_words: 词汇表大小
    :param atleast: 最小词频
    """
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    count = [['PAD', 0]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary





inputs = []
for sen in sentences:
    # replace the word with  corresponding id (one-hot representation)
    # 输入是将句子集合中的每个句子id化,作为训练数据集
    inputs.append(np.asarray([word_dict[n] for n in sen.split()[:1]]))

print(inputs)

import sys
sys.exit(0)
outputs = []
for out in labels:
    outputs.append(np.eye(num_classes)[out])  # ONE-HOT : To using Tensor Softmax Loss function

# Model
X = tf.placeholder(tf.int32, [None, sequence_length])  # sequence_length <=> step_size
Y = tf.placeholder(tf.int32, [None, num_classes])

# Embedding layer
W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedded_chars = tf.nn.embedding_lookup(W, X)  # [batch_size, sequence_length, embedding_size]
embedded_chars = tf.expand_dims(embedded_chars, -1)  # add channel(=1) [batch_size, sequence_length, embedding_size, 1]

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):  # filter_sizes = [2, 2, 2]  # n-gram window
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

    conv = tf.nn.conv2d(embedded_chars,  # [batch_size, sequence_length, embedding_size, 1]
                        W,  # [filter_size(n-gram window), embedding_size, 1, num_filters(=3)]
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    pooled = tf.nn.max_pool(h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            # [batch_size, filter_height, filter_width, channel]
                            strides=[1, 1, 1, 1],
                            padding='VALID')
    pooled_outputs.append(pooled)  # dim of pooled : [batch_size(=6), output_height(=1), output_width(=1), channel(=1)]

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs,
                   num_filters)  # h_pool : [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # [batch_size, ]

# Model-Training
Weight = tf.get_variable('W', shape=[num_filters_total, num_classes],
                         initializer=tf.contrib.layers.xavier_initializer())
Bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model = tf.nn.xw_plus_b(h_pool_flat, Weight, Bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Model-Predict
hypothesis = tf.nn.softmax(model)
predictions = tf.argmax(hypothesis, 1)
# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(len(inputs))
print(len(outputs))



for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: outputs})
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Test
test_text = 'sorry hate you'
tests = []
tests.append(np.asarray([word_dict[n] for n in test_text.split()]))

predict = sess.run([predictions], feed_dict={X: tests})
result = predict[0][0]
if result == 0:
    print(test_text, "is Bad Mean...")
else:
    print(test_text, "is Good Mean!!")

