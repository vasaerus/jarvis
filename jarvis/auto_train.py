import os
import sqlite3
from datetime import datetime
from transformers import pipeline, set_seed
from model import train_model
import openai
import os
# OpenAI API key
openai.api_key = "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"

jarvis_db = "jarvis.db"
jarvis_model = "jarvis.h5"

# Initialize the database and model if they don't exist
if not os.path.isfile(jarvis_db):
    conn = sqlite3.connect(jarvis_db)
    c = conn.cursor()
    c.execute('''CREATE TABLE conversations (id INTEGER PRIMARY KEY, prompt TEXT, response TEXT, date TEXT)''')
    conn.commit()
    conn.close()

if not os.path.isfile(jarvis_model):
    train_model()

# Initialize the AI pipeline
generator = pipeline('text-generation', model=jarvis_model, tokenizer='gpt2', device=0)

# Main function to run the auto train process
def auto_train():
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

    # Load the model
    model = load_model('jarvis.h5')

    # Train the model on new data
    model = train_model(model, input_data, output_data)

    # Save the trained model
    filepath = 'jarvis.h5'
    model.save(filepath)
    print("Jarvis has been successfully trained on new data.")

auto_train()

