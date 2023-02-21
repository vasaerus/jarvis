from mercury import *

# Set OpenAI API key
openai.api_key = "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"

def generate_online_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.9
    )

    generated_response = response.choices[0].text.strip()
    save_to_db(prompt, generated_response)

    return generated_response


def save_to_db(prompt, response):
    conn = sqlite3.connect('earth/jarvis.db')
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS chatbot_responses (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            response TEXT
        );
    """)

    # Insert the prompt and response
    c.execute("INSERT INTO chatbot_responses (prompt, response) VALUES (?, ?)", (prompt, response))

    conn.commit()
    conn.close()


def retrieve_response(prompt):
    conn = sqlite3.connect('earth/jarvis.db')
    c = conn.cursor()
    c.execute("SELECT response FROM chatbot_responses WHERE prompt = ?", (prompt,))
    response = c.fetchone()
    conn.close()
    if response is None:
        return None
    return response[0]

def get_model(num_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(num_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def save_model(max_seq_len=1000, vocab_size=1000, num_epochs=20, batch_size=32):
    max_seq_len = max_seq_len + 1
    # Load the conversation data from the database
    conn = sqlite3.connect('earth/jarvis.db')
    c = conn.cursor()
    conversations = c.execute("SELECT prompt, response FROM chatbot_responses").fetchall()
    conn.close()

    # Tokenize the conversation data
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([prompt for prompt, _ in conversations])
    sequences = tokenizer.texts_to_sequences([prompt for prompt, _ in conversations])

    # Save the tokenizer
    with open('earth/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

    # Set up the input and output data
    input_data = padded_sequences[:, :-1]
    output_data = np.zeros((len(sequences), vocab_size))
    for i, sequence in enumerate(padded_sequences):
        output_data[i, sequence[-1]] = 1

    # Load the Jarvis model
    model = get_model(vocab_size, max_seq_len)
    model = train_model(model, input_data, output_data, num_epochs=num_epochs, batch_size=batch_size)

    # Save the trained model
    model.save('earth/jarvis.h5')
    
def train_model(model, X_train, y_train, num_epochs=10, batch_size=32):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, verbose=1, batch_size=batch_size)
    return model

def generate_offline_response(prompt):
    max_seq_len = 1000  # Max sequence length for input sequences
    tokenizer_path = "earth/tokenizer.pkl"  # Path to tokenizer file
    model_path = "earth/jarvis.h5"  # Path to trained model file

    # Load the tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load the trained model
    if not os.path.isfile(model_path):
        raise ValueError(f"No model found at {model_path}")
    model = load_model(model_path)

    # Tokenize the input prompt
    input_sequence = tokenizer.texts_to_sequences([prompt])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_seq_len, padding="pre")

    # Generate the response using the model
    output_sequence = model.predict(padded_sequence)[0]
    output_token = np.argmax(output_sequence)
    output_word = tokenizer.index_word[output_token]

    return output_word

def select_mode():
    while True:
        print("\nSelect an option:")
        print("1. Online conversation")
        print("2. Offline conversation")
        print("3. Train model and save to jarvis.h5")
        print("4. Beta mode")
        print("5. Exit")

        choice = input()

        if choice in ["1", "2", "3", "4", "5"]:
            return choice

        print("Invalid choice. Please try again.")