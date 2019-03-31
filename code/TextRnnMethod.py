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

# get the training sentences
df_train = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/train.tsv', sep='\t')
df_test = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/test.tsv', sep='\t')

print(df_train)
phrase = df_train['Phrase'].tolist()

word_list = " ".join(phrase).split()

stop_word = set(stopwords.words('english'))

word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}

n_class = 5

# TextRNN Parameter
n_step = 2  # number of cells (or steps)
n_hidden = 5  # number of hidden units in one cell


def make_batch(phrase):
    """making training batches """
    input_batch = []
    target_batch = []

    for phr in phrase:
        word = phr.split()
        input = [word_dict[n] for n in word]
        target = word_dict[word]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch


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



