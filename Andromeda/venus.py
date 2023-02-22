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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_offline_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, do_sample=True)
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    save_to_db(prompt, generated_response)
    return generated_response

def select_mode():
    while True:
        print("\nSelect an option:")
        print("1. Jarvis : Online conversation")
        print("2. Jarvis : Offline conversation")
        print("3. * development * Train GPT2 model from Jarvis.db")
        print("4. Jarvis : using GPT2 model")
        print("5. Jarvis : using Jarvis.db")
        print("6. Exit")

        choice = input()

        if choice in ["1", "2", "3", "4", "5","6"]:
            return choice

        print("Invalid choice. Please try again.")