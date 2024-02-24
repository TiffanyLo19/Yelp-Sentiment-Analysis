# -*- coding: utf-8 -*-
"""Yelp Sentiment Analysis.ipynb

##### Sentiment Analysis for Yelp using LSTM and Naive Bayes

https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences#

### Pre-Processing
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import keras
import gensim
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import CountVectorizer

# Import dataset into pandas dataframe
reviews = pd.read_csv("https://raw.githubusercontent.com/TiffanyLo19/Yelp-Sentiment-Analysis/main/yelp_labelled.txt", sep ='delimiter', header = None, names = ["Review"])
reviews['Sentiment'] = reviews['Review'].str.strip().str[-1]
reviews['Review'] = reviews['Review'].str[:-2]
reviews

# Assign labels to sentiment
reviews['Sentiment'] = reviews.Sentiment.map({'1': 'positive', '0': 'negative'})
reviews

# Count of each sentiment
reviews.groupby('Sentiment').nunique()

# Check for null values
reviews = reviews[['Review','Sentiment']]
reviews["Review"].isnull().sum()

reviews.dtypes

# Create training and testing datasets
from sklearn.model_selection import train_test_split
X = reviews[['Review']]
Y = reviews[['Sentiment']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words = 'english')
for i in X_train:
  X_train[i] = vec.fit_transform(X_train[i]).toarray()
for i in X_test:
  X_test[i] = vec.fit_transform(X_test[i]).toarray()

# Check training and testing dataset shapes
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

"""### Sentiment Prediction Using LSTM"""

# Split the series into a list
temp = []
toList = reviews['Review'].values.tolist()
for i in range(len(toList)):
    temp.append(toList[i])
list(temp[:5])

# Remove punctuation
def toWords(sentences):
    for line in sentences:
        yield(gensim.utils.simple_preprocess(str(line), deacc = True))

sepWords = list(toWords(temp))

print(sepWords[:10])

# Detokenize
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

# Lowercase all sentences
words = []
for i in range(len(sepWords)):
    words.append(detokenize(sepWords[i]))
print(words[:10])

# Convert categorical to float type
labels = np.array(reviews['Sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'negative':
        y.append(0)
    if labels[i] == 'positive':
        y.append(1)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 2, dtype = "float32")
del y

len(labels)

max_words = 400
max_len = 100

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(words)
seq = tokenizer.texts_to_sequences(words)
words1 = pad_sequences(seq, maxlen = max_len)
print(words1)

#print(labels)

# Training and Testing using same ratios
X_train1, X_test1, y_train1, y_test1 = train_test_split(words1, labels, test_size = 0.2, random_state = 99)
print (len(X_train1), len(X_test1), len(y_train1), len(y_test1))

# Create and tune model
model = Sequential()
model.add(layers.Embedding(max_words, 40, input_length = max_len))
model.add(layers.Bidirectional(layers.LSTM(30, dropout = 0.6)))
model.add(layers.Dense(2, activation = 'softmax'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train1, y_train1, epochs = 40, validation_data = (X_test1, y_test1))

# Find overall model accuracy
test_loss, accuracy = model.evaluate(X_test1, y_test1, verbose = 2)
print('Model Accuracy: ', accuracy)

"""Sentence Testing"""

# Define sentiments
feeling = ['Negative', 'Positive']

seq = tokenizer.texts_to_sequences(['So dont go there if you are looking for good food...'])
test = pad_sequences(seq, maxlen = max_len)
feeling[np.around(model.predict(test)).argmax(axis = 1)[0]]

seq = tokenizer.texts_to_sequences(['If that bug never showed up I would have given a 4 for sure, but on the other side of the wall where this bug was climbing was the kitchen.'])
test = pad_sequences(seq, maxlen = max_len)
feeling[np.around(model.predict(test)).argmax(axis = 1)[0]]

seq = tokenizer.texts_to_sequences(['The warm beer didnt help.'])
test = pad_sequences(seq, maxlen = max_len)
feeling[np.around(model.predict(test)).argmax(axis = 1)[0]]

seq = tokenizer.texts_to_sequences(['The best place to go for a tasty bowl of Pho!'])
test = pad_sequences(seq, maxlen = max_len)
feeling[np.around(model.predict(test)).argmax(axis = 1)[0]]

"""### Sentiment Prediction Using Naive Bayes"""

# Import model and test hyperparamter
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=[0.1, 0.9])
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(priors=[0.5, 0.5])
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(var_smoothing = 1)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(priors=[0.95, 0.05])
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(var_smoothing = 100)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(var_smoothing = 0.0000001)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB(var_smoothing = 0)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Test unique hyperparamter combination
clf = GaussianNB()
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)

# Return 10 misclassified reviews
predictions = clf.predict(X_test)
Y_test.reset_index(inplace = True,drop = True)
X_test.reset_index(inplace = True,drop = True)
i = 0
j = 0
while (i < 10 and j < len(X_test)):
  if (predictions[j] != Y_test.iloc[j,0]):
      print(X_test2.iloc[j,0])
      i = i + 1
  j = j + 1
