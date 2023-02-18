import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

def train_model(model, X_train, y_train):
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, verbose=1, batch_size=64)
    
    return model
# Define hyperparameters
num_words = 1000
embedding_size = 100
max_sequence_len = 1000
units = 128
num_epochs = 10
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

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

# Set up the input and output data
input_data = padded_sequences[:, :-1]
output_data = np.zeros((len(sequences), num_words))
for i, sequence in enumerate(padded_sequences):
    output_data[i, sequence[-1]] = 1

# Train the model
filepath = 'jarvis.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(input_data, output_data, epochs=num_epochs, batch_size=batch_size, callbacks=[checkpoint])

# Save the trained model
model.save(filepath)
