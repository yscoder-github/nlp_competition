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
import numpy as np
from nltk.corpus import stopwords

__author__ = 'yscoder@foxmail.com'
tf.reset_default_graph()

# Data PreProcess
df_train = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/train.tsv', sep='\t')
df_test = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/test.tsv', sep='\t')

print(df_train)
phrase = df_train['Phrase'].tolist()
labels = df_train['Sentiment'].tolist()

word_list = " ".join(phrase).split()
word_list = list(set(word_list))
stop_word = set(stopwords.words('english'))

word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)



number_dict = {i: w for i, w in enumerate(word_list)}

# TextCNN Parameter
embedding_size = 2
sequence_length = 20
num_class = 5
filter_sizes = [2, 2, 2]
num_filters = 3




inputs = []
for phr in phrase:
    inputs.append(np.asarray([word_dict[n] for n in phr.split()]))
outputs = []
for out in labels:
    outputs.append(np.eye(num_class)[out])  # one-hot: To using Tensor softmax loss function





# Model
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, num_class])

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

embedding_chars = tf.nn.embedding_lookup(W, X)  # [batch_size, sequence_length, embedding_size]
embeddingg_chars = tf.expand_dims(embedding_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))

    conv = tf.nn.conv2d(embedding_chars,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    pooled = tf.nn.max_pool(h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID')

    pooled_outputs.append(pooled)

num_filters_total = num_filters * len(filter_size)
h_pool = tf.concat(pooled_outputs,
                   num_filters)








n_class = 5

# TextRNN Parameter
n_step = 2  # number of cells (or steps)
n_hidden = 5  # number of hidden units in one cell




# Model
X = tf.placeholder(tf.float32, [None, n_step, n_class])  # [batch_size, n_step, n_class]
Y = tf.placeholder(tf.float32, [None, n_class])  # [batch_size, n_class]

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# outputs : [batch_size, n_step, n_Hidden]
outputs = tf.transpose(outputs, [1, 0, 2])  # [n_step, batch_size, n_hidden]
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b  # [batch_size, n_class]

# optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)

# training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


input_batch, target_batch = make_batch(phrase)

for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1) % 1000 == 0:
        print('Epoch:{}, cost={:.6f}'.format(epoch + 1, loss))

# still have



























sentiment = df_train['Sentiment']


# print(phrase_column)



