from mercury import *
from venus import *

while True:
    choice = select_mode()

    if choice == "1":
        prompt = input("You: ")
        response = generate_online_response(prompt)
        print("Jarvis: " + response)

    elif choice == "2":
        prompt = input("Enter a prompt: ")
        response = input("Enter a response: ")
        save_to_db(prompt, response)
        print("Response saved to database.")

    elif choice == "3":
        func_bert()
        print("Let's train using BERT ")

    elif choice == "4":
        prompt = input("You: ")
        response = generate_offline_response(prompt, model, tokenizer)
        print("Jarvis: " + response)

    elif choice == "5":
        prompt = input("You: ")
        response = retrieve_response(prompt)
        if response:
            print("Jarvis: " + response)
        else:
            print("Jarvis: I don't have a response for that prompt.")
            response = input("Please enter a response for this prompt: ")
            save_to_db(prompt, response)

    elif choice == "6":
        prompt = input("Enter a prompt: ")
        image = generate_image_from_prompt(prompt)
        save_image_to_earth(image)

    elif choice == "7":
        break

    else:
        print("Invalid choice. Please try again.")
