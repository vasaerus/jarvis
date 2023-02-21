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
        save_model()
        print("Jarvis has been successfully trained on new data and saved to jarvis.h5.")

    elif choice == "4":
        prompt = input("You: ")
        response = generate_offline_response(prompt)
        print("Jarvis: " + response[0])

    elif choice == "5":
        break

    else:
        print("Invalid choice. Please try again.")