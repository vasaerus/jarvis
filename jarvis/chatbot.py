import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import openai
from model import max_sequence_len
import pickle 
from model import train_model

# Define hyperparameters
global num_words
num_words = 1000

# rest of the code

openai.api_key = "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"

conn = sqlite3.connect("jarvis.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS chatbot_responses (prompt text, response text)")
conn.commit()

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.9,
    ).choices[0].text
    return response

def save_response(prompt, response):
    c.execute("INSERT INTO chatbot_responses (prompt, response) VALUES (?, ?)", (prompt, response))
    conn.commit()

def retrieve_response(prompt):
    c.execute("SELECT response FROM chatbot_responses WHERE prompt=?", (prompt,))
    response = c.fetchone()
    if response:
        return response[0]
    else:
        return None

def select_mode():
    mode = input("Enter 'online' or 'offline': ")
    if mode not in ["online", "offline"]:
        print("Invalid mode. Please enter either 'online' or 'offline'.")
        return select_mode()
    return mode

def evaluate_response(response):
    satisfaction = input(f"Are you satisfied with the response '{response}'? Enter 'yes' or 'no': ")
    if satisfaction not in ["yes", "no"]:
        print("Invalid response. Please enter either 'yes' or 'no'.")
        return evaluate_response(response)
    return satisfaction

def get_online_response(prompt):
    return generate_response(prompt)

def save_to_db(prompt, response):
    c.execute("INSERT INTO chatbot_responses (prompt, response) VALUES (?, ?)", (prompt, response))
    conn.commit()

def update_model():
    pass

def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def generate_offline_response(prompt):
    # Load the tokenizer
    tokenizer = load_tokenizer()

    # Pad the sequence
    sequence = tokenizer.texts_to_sequences([prompt])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')

    # Load the Jarvis model
    model = load_model('jarvis.h5')
    
    # Generate a response using the model
    response = model.predict(sequence)
    response = np.argmax(response)
    if response in tokenizer.index_word:
        response = tokenizer.index_word[response]
        response = prompt + ' ' + response
    else:
        response = "I'm sorry, I don't know what to say."
    
    return response

def retrain_model():
    pass

def set_seed(seed):
    random.seed(seed)

def online_conversation():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        online_response = get_online_response(user_input)
        print("Jarvis: " + online_response)

def offline_conversation():
    while True:
        prompt = input("Enter a prompt: ")
        if prompt.lower() == "exit":
            break
        response = input("Enter a response: ")
        save_to_db(prompt, response)

def feedback_model():
    prompt = input("Enter a prompt for the model to respond to: ")
    response = generate_offline_response(prompt)
    satisfaction = evaluate_response(response)
    if satisfaction == "no":
        feedback = input("Please provide feedback to improve Jarvis: ")
        save_response(prompt, feedback)

def beta_conversation():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        online_response = get_online_response(user_input)
        satisfaction = evaluate_response(online_response)
        if satisfaction == "no":
            prompt = f"{user_input}\nJarvis: {online_response}\nYou:"
            feedback = input("Please provide feedback to improve Jarvis: ")
            save_response(prompt, feedback)

def run_bot():
    while True:
        print("\nSelect an option:")
        print("1. Online conversation")
        print("2. Offline conversation")
        print("3. Beta conversation with feedback")
        print("4. Feedback model")
        print("5. Auto train")
        print("6. Exit")

        choice = input()

        if choice == "1":
            prompt = input("You: ")
            response = generate_response(prompt)
            print("Jarvis: " + response)
        elif choice == "2":
            prompt = input("You: ")
            response = retrieve_response(prompt)
            if response:
                print("Jarvis: " + response)
            else:
                user_input = input("Jarvis: I'm sorry, I don't have a response for that. What would you like me to say? ")
                if user_input:
                    save_response(prompt, user_input)
                    print("Jarvis: " + user_input)
                else:
                    print("Jarvis: Okay, maybe next time.")
        elif choice == "3":
            beta_conversation()
        elif choice == "4":
            feedback_model()
        elif choice == "5":
            retrain_model()
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")
