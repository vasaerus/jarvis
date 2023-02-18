import openai
import sqlite3
import random
import os

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

mode = select_mode()
while True:
    prompt = input("Enter a prompt: ")
    if mode == "online":
        response = generate_response(prompt)
        save_response(prompt, response)
        print(response)
    else:
        response = retrieve_response(prompt)
        if response:
            print(response)
        else:
            response = generate_response(prompt)

def chat_loop():
  while True:
    # ask user for input
    user_input = input("You: ")

    # check if user wants to exit
    if user_input == "exit":
      break

    # check if the user wants to use the online API
    if user_input == "online":
      # call the online API to get the response
      response = get_online_response(user_input)

      # save the user input and response to the database
      save_to_db(user_input, response)

      # update the model with the latest data from the database
      update_model()

    # use the offline model to generate the response
    response = generate_offline_response(user_input)

    # ask the user to rate the response
    user_rating = input("How satisfied are you with the response? (1-5): ")

    # check if the user is satisfied with the response
    if int(user_rating) < 3:
      # the user is not satisfied, ask if they want to retrain
      retrain = input("Do you want to retrain the model? (yes/no): ")

      # check if the user wants to retrain
      if retrain == "yes":
        # retrain the model with the latest data from the database
        retrain_model()

    # print the response
    print("Bot: ", response)

# start the chat loop
chat_loop()
