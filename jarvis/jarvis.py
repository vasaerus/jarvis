import random
from keras.models import load_model
import nltk
import numpy as np
from utils.preprocess import preprocess_text

# Load the prompt and response data
prompts = []
responses = []
with open('data/prompts.csv', 'r') as f:
    for line in f:
        prompts.append(line.strip())
with open('data/responses.csv', 'r') as f:
    for line in f:
        responses.append(line.strip())

# Load the trained RNN model
model = load_model('models/rnn.h5')

# Set up NLTK tokenizer and stop words
tokenizer = nltk.tokenize.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Define function to generate a response to a prompt
def generate_response(prompt, model, tokenizer):
    # Preprocess the prompt
    prompt = preprocess_text(prompt, tokenizer, stop_words)

    # Convert the prompt to a sequence of word embeddings using the Word2Vec model
    # (not shown here, assuming the model has already been trained and saved)

    # Pad the sequence to a fixed length
    # (not shown here, assuming the sequence has already been padded to a fixed length)

    # Generate the response using the trained RNN model
    prompt_seq = np.array([prompt_seq])
    response_seq = model.predict(prompt_seq)[0]
    response = ' '.join([tokenizer.index_word[idx] for idx in response_seq])
    return response

# Main loop for interacting with the chatbot
while True:
    # Get user input
    user_input = input('You: ')

    # Check if user wants to exit
    if user_input.lower() in ['exit', 'quit']:
        break

    # Generate a random prompt from the available prompts
    prompt = random.choice(prompts)

    # Generate a response to the prompt
    response = generate_response(prompt, model, tokenizer)

    # Print the response
    print(f"Jarvis: {response}")
