import os
import sqlite3
import openai
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

openai.api_key = "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"

jarvis_db = "jarvis.db"
jarvis_model = "jarvis.h5"

if not os.path.isfile(jarvis_db):
    conn = sqlite3.connect(jarvis_db)
    c = conn.cursor()
    c.execute('''CREATE TABLE conversations (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT, image_file TEXT, audio_file TEXT, video_file TEXT, date TEXT, trained INT DEFAULT 0)''')
    conn.commit()
    conn.close()

# Define hyperparameters
num_words = 10000
embedding_size = 100
max_sequence_len = 20
units = 128
num_epochs = 20
batch_size = 32

# Load the Jarvis model
model = Sequential()
model.add(Embedding(num_words, embedding_size, input_length=max_sequence_len-1))
model.add(LSTM(units))
model.add(Dropout(0.2))
model.add(Dense(num_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Load the conversation data from the database
conn = sqlite3.connect('jarvis.db')
c = conn.cursor()
conversations = c.execute("SELECT prompt, response FROM chatbot_responses").fetchall()
conn.close()

# Tokenize the conversation data
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts([prompt for prompt, _ in conversations])
sequences = tokenizer.texts_to_sequences([prompt for prompt, _ in conversations])

# Load new conversation data
new_data = [("What is your favorite food?", "I like pizza."),
            ("Do you have any pets?", "No, I don't have any pets."),
            ("What is your favorite color?", "My favorite color is blue.")]

# Tokenize the new conversation data
new_sequences = tokenizer.texts_to_sequences([prompt for prompt, _ in new_data])
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_len, padding='pre')

# Combine the old and new conversation data
padded_sequences = np.concatenate((padded_sequences, new_padded_sequences), axis=0)

# Set up the input and output data
input_data = padded_sequences[:, :-1]
output_data = np.zeros((len(padded_sequences), num_words))
for i, sequence in enumerate(padded_sequences):
    output_data[i, sequence[-1]] = 1

# Train the model
filepath = 'jarvis.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(input_data, output_data, epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint])

# Save the trained model
model.save(filepath)

print("Jarvis has been successfully trained on new data.")