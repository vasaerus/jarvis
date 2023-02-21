import os
import sqlite3
import argparse
from gensim.models import Word2Vec, KeyedVectors
from train import train_word2vec_model, train_chatbot_model
from chatbot import generate_response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1, help='mode: 1=train word2vec model, 2=train chatbot model, 3=generate chatbot response, 4=exit')
    parser.add_argument('--model_dir', type=str, default='models', help='path to the directory containing the word2vec model')
    parser.add_argument('--word2vec_model_file', type=str, default='word2vec_model.h5', help='name of the word2vec model file')
    parser.add_argument('--chatbot_model_file', type=str, default='jarvis.h5', help='name of the chatbot model file')
    parser.add_argument('--max_vocab_size', type=int, default=10000, help='maximum size of the vocabulary')
    parser.add_argument('--max_length', type=int, default=100, help='maximum length of the input/output sequences')
    args = parser.parse_args()

    if args.mode == 1:
        train_word2vec_model(args.model_dir, args.word2vec_model_file)
    elif args.mode == 2:
        train_chatbot_model(args.model_dir, args.word2vec_model_file, args.chatbot_model_file, args.max_vocab_size, args.max_length)
    elif args.mode == 3:
        word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(args.model_dir, args.word2vec_model_file), binary=False)
        chatbot_model = load_model(os.path.join(args.model_dir, args.chatbot_model_file))
        while True:
            prompt = input('You: ')
            if prompt == 'exit':
                break
            response = generate_response(prompt, word2vec_model, chatbot_model, args.max_length)
            print(f'Jarvis: {response}')
    elif args.mode == 4:
        exit()
