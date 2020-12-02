# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:39:04 2020

@author: anish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense,Dropout,SimpleRNN,LSTM,Conv1D,MaxPooling1D,GlobalMaxPooling1D,GRU
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
import statistics 


df = pd.read_csv("Codalab-train-dataset.csv")
texts = df.iloc[:, 1]
labels = df.iloc[:, 4]

# texts, test_examples, labels, test_labels = train_test_split(X,Y, test_size=0.2,random_state=0)

# Tokenizing the data
maxlen = 100 # cut off sentences after 50 words
max_words = 10000 # only consider top 10000 common words in dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens'%len(word_index))

# padding the sequences
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.array(labels)

print('Shape of data tensor:',data.shape)
print('Shape of labels tensor:',labels.shape)

# shuffle the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


embeddings_index = {}

f = open('glove.6B.100d.txt',"r", encoding="utf-8",)
for line in f:
     values = line.split()
     word = values[0]
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
f.close()

print('Found %s word vectors:'%len(embeddings_index))

# preparing glove word embeddings matrix
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
  if i<max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector # for words not in embedding index values will be zeros
      
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) # adding regularization

model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics = ['acc', f1])
history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)

def plot_result(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  f1 = history.history['f1']
  val_f1 = history.history['val_f1']
  
  epochs = range(1, len(acc)+1)

  plt.plot(epochs, acc, label='Training acc')
  plt.plot(epochs, val_acc, label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('epochs')
  plt.ylabel('acc')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, label='Training loss')
  plt.plot(epochs, val_loss, label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()

  plt.figure()

  plt.plot(epochs, f1, label='Training fmeasure')
  plt.plot(epochs, val_f1, label='Validation fmeasure')
  plt.title('Training and validation fmeasure')
  plt.xlabel('epochs')
  plt.ylabel('f1')
  plt.legend()
  
  plt.show()
  
  
print("Accuracy", statistics.mean(history.history['acc']))
test_dataset = pd.read_csv("Codalab-test-dataset.csv")
test_examples = test_dataset.iloc[:, 1]
tokenizer.fit_on_texts(test_examples)

sequences_test = tokenizer.texts_to_sequences(test_examples)

# padding the sequences
data_test = pad_sequences(sequences_test, maxlen=maxlen)
test_labels = model.predict(data_test)
# plotting the results
plot_result(history)


# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())

# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3)) # adding regularization

# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics = ['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)

# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))
# model.add(SimpleRNN(64, dropout=0.1))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data,labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))
# model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))

# model.add(Conv1D(32, 5, activation='relu'))
# model.add(MaxPooling1D(5))

# model.add(Conv1D(64, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())

# model.add(Dense(1, activation='sigmoid'))

# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))

# model.add(Conv1D(32, 5, activation='relu'))
# model.add(MaxPooling1D(5))

# model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.5))

# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))
# model.add(Conv1D(32, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)

# model = Sequential()
# model.add(Embedding(max_words, 32))
# model.add(Conv1D(32, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(64, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(GRU(32, activation='relu', dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
# model.add(GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
# history = model.fit(data, labels, epochs=20, batch_size=32, validation_split=0.1)
# print("Accuracy", statistics.mean(history.history['acc']))
# plot_result(history)