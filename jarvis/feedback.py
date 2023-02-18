import sqlite3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Jarvis database
jarvis_db = "jarvis.db"

# Jarvis model
jarvis_model = "jarvis.h5"

# Tokenizer file
tokenizer_file = "tokenizer.pickle"

def feedback_model(prompt, response, rating):
    if rating == 4 or rating == 5:
        # Concatenate the prompt and response
        text = prompt + "\nJarvis: " + response

        # Load the saved tokenizer
        with open(tokenizer_file, "rb") as f:
            tokenizer = pickle.load(f)

        # Tokenize the text
        sequences = tokenizer.texts_to_sequences([text])

        # Pad the sequences
        sequences_padded = pad_sequences(sequences, maxlen=40, padding='post', truncating='post')

        # Load the Jarvis model
        model = load_model(jarvis_model)

        # Generate the response from the model
        response_seq = model.predict(sequences_padded)[0]
        response_text = tokenizer.sequences_to_texts([response_seq])[0]

        # Store the newly trained model
        conn = sqlite3.connect(jarvis_db)
        c = conn.cursor()
        c.execute("INSERT INTO models (model, date) VALUES (?, datetime('now'))", (response_text, ))
        conn.commit()
        conn.close()

        print("Jarvis: Thank you for your feedback. I've updated my responses.")
    else:
        print("Jarvis: I'm sorry to hear that. I'll try to do better next time.")
