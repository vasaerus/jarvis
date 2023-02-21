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

    # Pad the index sequences to a fixed length
    input_seqs = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant', constant_values=0) for seq in input_seqs])
    output_seqs = np.array([np.pad(seq, (0, max_length - len(seq)), mode='constant', constant_values=0) for seq in output_seqs])

    return prompt_seqs, response_seqs, input_vocabulary, output_vocabulary, input_word_to_index, input_index_to_word, output_word_to_index, output_index_to_word, input_seqs, output_seqs
