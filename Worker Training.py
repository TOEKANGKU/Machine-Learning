!pip install nltk
!pip install Sastrawi

import string
import numpy as np
import pandas as pd
import csv
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.layers import Bidirectional, LSTM
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Loading the dataset
workers = pd.read_csv('/content/gdrive/MyDrive/data worker/dataset_worker.csv')

# Counting the rows for each label
workers['label'].value_counts()

# Wordcount for each text row and counting the largest length
workers['word_length'] = workers['text'].str.split()
workers['word_length'] = workers['word_length'].str.len()

workers['word_length'].sort_values(ascending=False)

# Function to remove punctuation
def remove_punctuation(sentences):
  translator = str.maketrans('', '', string.punctuation)
  no_punct = sentences.translate(translator)

  return no_punct

# Function to lowercase sentences and remove stopwords
def remove_stopword(sentences):
  # Lowercase and remove punctuation for sentences in each row
  sentences = sentences.lower()
  sentences = remove_punctuation(sentences)

  # Remove stopword
  factory = StopWordRemoverFactory()
  stopwords = factory.get_stop_words()

  words = sentences.split()
  words_result = [word for word in words if word not in stopwords]

  sentences = ' '.join(words_result)

  return sentences

# Function to parse the csv data into an array of sentences and labels
def parse_data(filename):

  sentences = []
  labels = []

  with open (filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter =',')

    next(reader, None)

    for row in reader:
      labels.append(remove_stopword(row[1]))
      sentences.append(remove_stopword(row[0]))

  return sentences, labels

# Parsing the data and checking the row count
sentences, labels = parse_data("/content/gdrive/MyDrive/data worker/dataset_worker.csv")
print("ORIGINAL DATASET:\n")
print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}\n\n")

# Variables for tokenizing
vocab_size = 10000
embedding_dim = 16
max_length = 100
oov_token = "<OOV>"
trunc_type = 'post'
training_size = 0.9

# Tokenizing the sentences and labels
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(sentences)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type, padding = 'post')
print('Shape of padded tensor:',padded.shape)
print(padded)

labels = pd.get_dummies(workers["label"]).values
print('Shape of label tensor:',labels.shape)
cek_label = pd.get_dummies(workers["label"])
print(cek_label.head())
print(labels)

#Spliting the data for training dan testing
random_seed = 42
train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(
    padded, labels, test_size=1-training_size, random_state=random_seed
)

# Checking the sentences count for training and test
jml_train = len(train_sentences)
jml_test = len(validation_sentences)

print(f"There are {jml_train} sentences for training and {jml_test} for testing.")

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(64, 5, activation = 'relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(15, activation = 'softmax')
])

model.summary()

# Compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(
    optimizer = optimizer,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(
          train_sentences,
          train_labels,
          validation_data = (validation_sentences, validation_labels),
          epochs = 400,
          verbose = 2
      )

# Saving the model
model = model_worker()
model.save("model_worker.h5")

# Rounding the testing labels
rounded_labels = np.argmax(validation_labels, axis=1)

# Rounding the prediction labels
predictions = model.predict(validation_sentences)
predicted_labels = np.argmax(predictions, axis=1)

# Checking the label class
unique_classes = np.unique(rounded_labels)
print("Unique Predicted Classes:", unique_classes)
print("Number of Unique Predicted Classes:", len(unique_classes))

unique_class = np.unique(predicted_labels)
print("Unique Predicted Classes:", unique_class)
print("Number of Unique Predicted Classes:", len(unique_class))


# Plotting loss and accuracy
plt.title('Loss')
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

# Input new text (pengalaman)
pengalaman = 'saya telah bekerja sebagai pengukir selama beberapa tahun. saya memiliki banyak pengalaman bekerja membuat ukiran kayu untuk dipasang pada rumah tradisional bali.'

# Text preprocessing for pengalaman
pengalaman = pengalaman.lower()

translator = str.maketrans('', '', string.punctuation)
pengalaman = pengalaman.translate(translator)

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

kata = pengalaman.split()
hasil_kata = [word for word in kata if word not in stopwords]

pengalaman = ' '.join(hasil_kata)

pengalaman = [f'{pengalaman}']

sequence = tokenizer.texts_to_sequences(pengalaman)
padded_sequence = pad_sequences(sequence, maxlen=max_length)

# Performing inference (predicting the class)
predict = model.predict(padded_sequence)
predict_label = labels[np.argmax(predict)]
labels = ['service ac', 'service kulkas', 'service mesin cuci', 'service soundsystem', 'service televisi', 'service data', 'service keamanan', 'service perangkat keras', 'service perangkat lunak', 'service sistem', 'tukang batu', 'tukang cat', 'tukang kayu', 'tukang las', 'tukang semen']

# Print prediction result
print(predict)
print(predict_label)

#Confusion matrix
confusion_matrix(rounded_labels, predicted_labels)

true_labels = unique_labels(rounded_labels)
pred_labels = unique_labels(predicted_labels)

def plot(y_true, y_pred):
  labels = true_labels
  column = [f'Predicted{label}' for label in labels]
  indicies = [f'Actual{label}' for label in labels]
  table = pd.DataFrame(confusion_matrix(y_true, y_pred),
                       columns=column, index=indicies)

  return table

plot(rounded_labels, predicted_labels)
