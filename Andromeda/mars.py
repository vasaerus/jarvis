from mercury import *

def main():
    # Set up logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set up the paths to the models and data
    models_dir = 'models/'
    data_path = os.path.join(models_dir, 'text8')
    model_path = os.path.join(models_dir, 'english.bin')
    output_path = 'models/english.txt'

    # Load the data
    sentences = gensim.models.word2vec.LineSentence(data_path)

    # Train the Word2Vec model
    model = Word2Vec(sentences, vector_size=100000, window=5, min_count=5, workers=8, sg=0, epochs=50)

    # Save the model in binary format
    model.save(model_path)

    # Save the model in text format
    model.wv.save_word2vec_format(output_path, binary=False)

if __name__ == '__main__':
    main()
