# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
"""
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
import pandas as pd
from keras import regularizers
from keras.layers import Input, Embedding, Dense, LSTM, Dropout
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/Sentiment Analysis on Movie Reviews/train.tsv", sep="\t")
data_test = pd.read_csv("../input/Sentiment Analysis on Movie Reviews/test.tsv", sep="\t")

X_train = data_train['Phrase']
Y_train = pd.Series(data_train['Sentiment']).values

X_test = data_test['Phrase']

Y_train = Y_train.reshape([-1, 1])
one_hot = OneHotEncoder(sparse=False)
Y_oneh = one_hot.fit_transform(Y_train)

words_list = []
for i in range(X_train.shape[0]):
    t_list = X_train[i].split(" ")
    for each_word in t_list:
        words_list.append(each_word)

for i in range(X_test.shape[0]):
    t_list = X_test[i].split(" ")
    for each_word in t_list:
        words_list.append(each_word)

vocab_list = list(set(words_list))

words_indices = {}
indices_words = {}
for i, word in enumerate(vocab_list):
    words_indices[word] = i
    indices_words[i] = word

X_train_n = np.zeros((X_train.shape[0], 100))
for i in range(X_train.shape[0]):
    words = X_train[i].split(" ")
    for j, each_w in enumerate(words):
        X_train_n[i, j] = words_indices[each_w]

in_M = Input(shape=(100,))
M = Embedding(input_dim=len(vocab_list), output_dim=300)(in_M)
M = LSTM(128, return_sequences=True)(M)
M = Dropout(0.2)(M)
M = LSTM(128, return_sequences=False)(M)
M = Dropout(0.2)(M)
M_out = Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.1))(M)

model = Model(inputs=in_M, outputs=M_out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_n, Y_oneh, epochs=10, batch_size=64)

X_test_n = np.zeros((X_test.shape[0], 100))
for i in range(X_test.shape[0]):
    words = X_test[i].split(" ")
    for j, each_w in enumerate(words):
        X_test_n[i, j] = words_indices[each_w]
y_test = model.predict(X_test_n)
