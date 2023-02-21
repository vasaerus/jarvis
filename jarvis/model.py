import os
import numpy as np
import nltk
import gensim
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import sqlite3
import csv
from sklearn.model_selection import train_test_split


def preprocess_text(text, tokenizer, stop_words):
    # Convert the text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Remove stop words from the text
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join the filtered tokens back into a string
    processed_text = ' '.join(filtered_tokens)

    return processed_text


def preprocess(max_length, word2vec_model, prompt_response_pairs, max_vocab_size):
    # Set up NLTK tokenizer and stop words
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Preprocess the text data
    preprocessed_prompts = [preprocess_text(prompt, tokenizer, stop_words) for prompt, response in prompt_response_pairs]
    preprocessed_responses = [preprocess_text(response, tokenizer, stop_words) for prompt, response in prompt_response_pairs]

    # Convert the preprocessed text data to sequences of word embeddings
    prompt_seqs = [[word2vec_model.wv[word] for word in prompt.split() if word in word2vec_model.wv] for prompt in preprocessed_prompts]
    response_seqs = [[word2vec_model.wv[word] for word in response.split() if word in word2vec_model.wv] for response in preprocessed_responses]

    # Pad the sequences to a fixed length
    pad_vector = np.zeros(word2vec_model.vector_size)
    prompt_seqs = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=pad_vector) for seq in prompt_seqs])
    response_seqs = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant', constant_values=pad_vector) for seq in response_seqs])

    # Create the input/output vocabularies
    input_vocabulary = set([word for prompt in preprocessed_prompts for word in prompt.split()])
    output_vocabulary = set([word for response in preprocessed_responses for word in response.split()])

    # Limit the size of the input and output vocabularies
    input_vocabulary = sorted(list(input_vocabulary))[:max_vocab_size]
    output_vocabulary = sorted(list(output_vocabulary))[:max_vocab_size]

    # Create word-to-index and index-to-word mappings for the input and output vocabularies
    input_word_to_index = {word: i for i, word in enumerate(input_vocabulary)}
    input_index_to_word = {i: word for i, word in enumerate(input_vocabulary)}
    output_word_to_index = {word: i for i, word in enumerate(output_vocabulary)}
    output_index_to_word = {i: word for i, word in enumerate(output_vocabulary)}

    # Convert the input and output sequences to index sequences
    input_seqs = [[input_word_to_index[word] for word in prompt.split() if word in input_vocabulary] for prompt in preprocessed_prompts]
    output_seqs = [[output_word_to_index[word] for word in response.split() if word in output_vocabulary] for response in preprocessed_responses]

   

def train_model(max_length, word2vec_model, model_file):
# Connect to the SQLite database
conn = sqlite3.connect('jarvis.db')
c = conn.cursor()
# Load the prompt and response data from the database
c.execute('SELECT prompt, response FROM chatbot_responses')
data = c.fetchall()
prompt = [row[0] for row in data]
response = [row[1] for row in data]

# Split the data into training and validation sets
train_prompt, val_prompt, train_response, val_response = train_test_split(prompt, response, test_size=0.1)

# Preprocess the data and convert it to sequences
train_prompt_seqs, train_response_seqs, input_vocabulary, output_vocabulary, input_word_to_index, input_index_to_word, output_word_to_index, output_index_to_word, train_input_seqs, train_output_seqs = preprocess(max_length, word2vec_model, list(zip(train_prompt, train_response)), 10000)
val_prompt_seqs, val_response_seqs, _, _, _, _, _, _, val_input_seqs, val_output_seqs = preprocess(max_length, word2vec_model, list(zip(val_prompt, val_response)), 10000)

# Define the RNN model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, word2vec_model.vector_size)))
model.add(Dense(64, activation='relu'))
model.add(Dense(max_length, activation='softmax'))

# Define the optimizer and compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define a callback to save the model weights after each epoch
checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the model
model.fit(train_input_seqs, train_output_seqs, validation_data=(val_input_seqs, val_output_seqs), epochs=50, batch_size=64, callbacks=callbacks_list, verbose=1)

if name == 'main':
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    # Load each model, train the RNN model, and save the trained weights
for model_file in os.listdir(model_dir):
    print(f'Training RNN model using {model_file}...')
    word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(model_dir, model_file), binary=False)
    model_file_path = os.path.join(model_dir, f'rnn_model_{model_file}.h5')
    train_model(100, word2vec_model, model_file_path
