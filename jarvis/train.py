import os
from gensim.models import KeyedVectors

model_dir = 'models/'

# Get a list of all the files in the directory
model_files = os.listdir(model_dir)

# Loop through each file and load the model
for model_file in model_files:
    print(f'Loading model {model_file}...')
    word_vectors = KeyedVectors.load_word2vec_format(os.path.join(model_dir, model_file), binary=False)
    
for model_file, model_name in models:
    print(f'Loading model {model_name}...')
    word_vectors = KeyedVectors.load_word2vec_format(model_file, binary=False)

    print(f'Evaluating model {model_name}...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(100, 300)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    prompt_seqs, response_seqs, input_vocabulary, output_vocabulary, input_word_to_index, input_index_to_word, output_word_to_index, output_index_to_word, input_seqs, output_seqs = preprocess(100, word_vectors, prompt_response_pairs, 10000)

    model.load_weights(f'{model_name}.h5')
    _, accuracy = model.evaluate(prompt_seqs, response_seqs, verbose=0)
    print(f'Accuracy for model {model_name}: {accuracy}')

# Select one of the models to use for training
word_vectors = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False)

def train_model(max_length, word2vec_model):
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