import nltk

# Set up NLTK tokenizer and stop words
tokenizer = nltk.tokenize.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text, tokenizer, stop_words):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text into individual words
    tokens = tokenizer.tokenize(text)

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Remove non-alphanumeric characters
    filtered_tokens = [token for token in filtered_tokens if token.isalnum()]

    # Join filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text
