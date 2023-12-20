!pip install nltk
!pip install Sastrawi

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def model_user():
  # Loading the datasets
  user = pd.read_csv('/content/gdrive/MyDrive/data user/dataset_user.csv')
  user

  #Counting the rows for each label
  user['Kategori Pekerjaan'].value_counts()

  # Variables for tokenizing
  vocab_size = 10000
  embedding_dim = 16
  max_length = 100
  oov_token = "<OOV>"
  trunc_type = 'post'
  training_size = .9

  # Function to remove punctuation
  def remove_punc(sentences):
    translator = str.maketrans('', '', string.punctuation)
    no_punct = sentences.translate(translator)

    return no_punct

  # Function to remove stopwords and make the sentences in lowercase
  def remove_stopwords(sentences):
    # Making the sentences in lowercase
    sentences = sentences.lower()
    sentences = remove_punc(sentences)

    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()

    words = sentences.split()
    words_result = [word for word in words if word not in stopwords]

    sentences = ' '.join(words_result)

    return sentences

  # Function to parse the csv datasets into arrays of sentences and labels
  def parse_data(filename):

    sentences = []
    labels = []

    with open (filename, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter =',')

      next(reader, None)

      for row in reader:
        labels.append(remove_stopwords(row[1]))
        sentences.append(remove_stopwords(row[0]))

      return sentences, labels
    
  import string

  # Parsing the data and checking the row count
  sentences, labels = parse_data("/content/gdrive/MyDrive/data user/dataset_user.csv")
  print("ORIGINAL DATASET:\n")
  print(f"There are {len(sentences)} sentences in the dataset.\n")
  print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
  print(f"There are {len(labels)} labels in the dataset.\n")
  print(f"The first 5 labels are {labels[:5]}\n\n")

  # Tokenizing the sentences
  tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
  tokenizer.fit_on_texts(sentences)

  # Tokenizing the labels
  label_tokenizer = Tokenizer()
  label_tokenizer.fit_on_texts(labels)

  sequences = tokenizer.texts_to_sequences(sentences)
  padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type, padding = 'post')

  labels = pd.get_dummies(user["Kategori Pekerjaan"]).values
  print('Shape of label tensor:',labels.shape)

  cek_label = pd.get_dummies(user["Kategori Pekerjaan"])
  print(cek_label.head())

  # Splitting the data for training and testing
  random_seed = 42
  train_sentences, test_sentences, train_labels, test_labels = train_test_split(
      padded, labels, test_size = 1-training_size, random_state = random_seed
  )

  jml_train = len(train_sentences)
  print(jml_train)

  # CNN model
  model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Conv1D(64, 5, activation = 'relu'),
          tf.keras.layers.GlobalMaxPooling1D(),
          tf.keras.layers.Dense(15, activation = 'softmax')
      ])

  # Compile and training the model
  model.summary()

  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
  model.compile(
      optimizer = optimizer,
      loss = 'categorical_crossentropy',
      metrics = ['accuracy']
  )

  history = model.fit(
      train_sentences,
      train_labels,
      validation_data = (test_sentences, test_labels),
      epochs = 400,
      verbose = 2
  )
  return model

# Saving the model
model = model_user()
model.save("model_user.h5")