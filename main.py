import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import speech_recognition as sr
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the dataset (assumes you have audio files in the "data" directory)
data_dir = "data"
labels = os.listdir(data_dir)
labels_to_index = {label: i for i, label in enumerate(labels)}

data = []
for label in labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        file_path = os.path.join(label_dir, filename)
        data.append([file_path, labels_to_index[label]])

df = pd.DataFrame(data, columns=["file_path", "label"])

# Use SpeechRecognition to convert audio to text
recognizer = sr.Recognizer()

def audio_to_text(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return ""

df["text"] = df["file_path"].apply(audio_to_text)

# Explore and preprocess the data
print(df.head())
print(df["label"].value_counts())

# Visualize the data
sns.countplot(x="label", data=df)
plt.show()

# Split the data
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the text data (convert to sequences, pad)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max([len(seq) for seq in X_train])
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Build a simple CNN model for text classification
num_classes = len(labels)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(max_sequence_length, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
y_train = to_categorical(y_train, num_classes)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_test = to_categorical(y_test, num_classes)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
