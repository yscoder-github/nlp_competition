# The code below will give you simple way to run/understand Keras Long-Short Term memory solution (LSTM) on given dataset.
# and same done ussing Machine Learning(RandomForest, Naive bayes and SVM) techniques.
# Later you can compare results and decide which solution to use.
# GOOD LUCK
import re

import nltk
import pandas as pd
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# git hub https://github.com/Stass88/lattelecom
df_train = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/train.tsv', sep='\t')
df_test = pd.read_csv('../input/Sentiment Analysis on Movie Reviews/test.tsv', sep='\t')

# this should help you to decide whether to use STOP WORDS or not.
# This part of code is just great analytical tool
stop_word = set(stopwords.words('english'))
word_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', min_df=0.001)
sparse_matrix = word_vectorizer.fit_transform(df_test['Phrase'])
frequencies = sum(sparse_matrix).toarray()[0]
freq = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
freq.sort_values('frequency', ascending=False)

# Visualization of data set
a = df_train.Sentiment.value_counts()
a = pd.DataFrame(a)
a['Rating'] = a.index
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(y='Sentiment', x='Rating', data=a)

# data preproccesing
# we make text lower case and leave only letters from a-z and digits
df_train['Phrase'] = df_train['Phrase'].str.lower()
df_train['Phrase'] = df_train['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
df_test['Phrase'] = df_test['Phrase'].str.lower()
df_test['Phrase'] = df_test['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

X_train = df_train.Phrase
y_train = df_train.Sentiment
tokenize = Tokenizer()
tokenize.fit_on_texts(X_train.values)

X_test = df_test.Phrase
X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)

max_lenght = max([len(s.split()) for s in df_train['Phrase']])
X_train = pad_sequences(X_train, max_lenght)
X_test = pad_sequences(X_test, max_lenght)

print(X_train.shape)
print(X_test.shape)

# Model building
# I choose to build 3 hidden layers
EMBEDDING_DIM = 100
unknown = len(tokenize.word_index) + 1
model = Sequential()
model.add(Embedding(unknown, EMBEDDING_DIM, input_length=max_lenght))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

print(y_train.shape)
print(X_train.shape)

model.fit(X_train, y_train, batch_size=128, epochs=7, verbose=1)

final_pred = model.predict_classes(X_test)

final_pred = pd.read_csv(r'../input/sampleSubmission.csv', sep=',')
final_pred.Sentiment = final_pred
final_pred.to_csv(r'results.csv', sep=',', index=False)
# The best result I had was around 0.66.. Results tested on test data


# Additionally before I took deep learning technique, I tested Naive Bayes,
# Random Forest and SVM approach to test which model works better.
# RF showed the best results for test data 0.62.
# Code below
# As well I tried to use
df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_train['Phrase'] = df_train['Phrase'].str.lower()
stop_word = set(stopwords.words('english'))
# df_train['Phrase_no_stopwords'] = df_train['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_word)]))
df_train['tokezines_sents'] = df_train.apply(lambda x: nltk.word_tokenize(x['Phrase']), axis=1)
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
df_train['tokezines_sents'] = df_train['tokezines_sents'].apply(lambda x: [stemmer.stem(y) for y in x])
df_train['tokezines_sents'] = df_train['tokezines_sents'].apply(lambda x: ' '.join(x))
from sklearn.model_selection import train_test_split

x = df_train.tokezines_sents
y = df_train.Sentiment
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train_df = vect.fit_transform(X_train)
x_test_df = vect.transform(X_test)
print('Number of features:', len(vect.get_feature_names()))
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(x_train_df, y_train)
y_pred_class = nb.predict(x_test_df)
print('NB:', metrics.accuracy_score(y_test, y_pred_class))
from sklearn.linear_model import SGDClassifier

SVM = SGDClassifier()
SVM.fit(x_train_df, y_train)
y_pred_class = SVM.predict(x_test_df)
print('SVM:', metrics.accuracy_score(y_test, y_pred_class))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train_df, y_train)
y_pred_class = rfc.predict(x_test_df)
print('RF:', metrics.accuracy_score(y_test, y_pred_class))
