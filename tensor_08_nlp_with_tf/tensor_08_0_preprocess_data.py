import tensorflow as tf
from keras.layers import TextVectorization, Embedding

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import random
import wget

# Import helper functions
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_histories


class PreprocessData:

    def __init__(self, nlp_dataset_path='datasets\\nlp_dataset\\'):
        self.train_df = pd.read_csv(nlp_dataset_path+'train.csv')
        self.test_df = pd.read_csv(nlp_dataset_path+'test.csv')

        self.train_df_shuffled = self.train_df.sample(frac=1, random_state=42)

        self.train_sentences = None
        self.train_labels = None
        self.val_sentences = None
        self.val_labels = None

        self.max_vocab_length = 10000
        self.max_output_sequence_length = 15

    def get_train_val_data(self):

        self.train_sentences, self.val_sentences, self.train_labels, self.val_labels = train_test_split(self.train_df_shuffled['text'].to_numpy(),
                                                                                                        self.train_df_shuffled['target'].to_numpy(),
                                                                                                        test_size=0.1,
                                                                                                        random_state=42)
        return self.train_sentences, self.train_labels, self.val_sentences, self.val_labels

    def get_test_dataframe(self):

        return self.test_df


def run():
    """08.0 Preprocess data for NLP"""

    '''Download a text dataset'''

    # download data (same as from Kaggle)
    # wget.download("https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip")

    # Unzip data
    # unzip_data('nlp_getting_started.zip')

    '''Visualizing a text dataset'''

    # Turn .csv files into pandas DataFrame's
    nlp_dataset_path = 'datasets\\nlp_dataset\\'
    train_df = pd.read_csv(nlp_dataset_path + 'train.csv')
    test_df = pd.read_csv(nlp_dataset_path + 'test.csv')
    # print(train_df.head())

    # Shuffle training dataframe
    train_df_shuffled = train_df.sample(frac=1, random_state=42)  # shuffle with random_state=42 for reproducibility
    # print(train_df_shuffled.head())

    # The test data doesn't have a target (that's what we'd try to predict)
    # print(test_df.head())

    # How many examples of each class?
    # print(train_df.target.value_counts())

    # How many samples total?
    # print(f"Total training samples: {len(train_df)}")
    # print(f"Total test samples: {len(test_df)}")
    # print(f"Total samples: {len(train_df) + len(test_df)}")

    # Let's visualize some random training examples
    # random_index = random.randint(0, len(train_df)-5)  # create random indexes not higher than the total number
    # for row in train_df_shuffled[['text', 'target']][random_index:random_index+5].itertuples():
    #     _, text, target = row
    #     print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
    #     print(f"Text:\n{text}\n")
    #     print("---\n")

    '''Split data into training and validation sets'''

    # Use train_test_split to split training data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled['text'].to_numpy(),
                                                                                train_df_shuffled['target'].to_numpy(),
                                                                                test_size=0.1,  # dedicate 10% of samples to validation set
                                                                                random_state=42)  # random state for reproducibility

    # Check the lengths
    # print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))

    # View the first 10 training sentences and their labels
    # print(train_sentences[:10], train_labels[:10])

    '''Converting text into numbers'''

    # Use the default TextVectorization variables
    # text_vectorizer = TextVectorization(max_tokens=None,  # how many words in the vocabulary
    #                                                       # (all the different words in your text)
    #                                     standardize="lower_and_strip_punctuation",  # how to process text
    #                                     split="whitespace",  # how to split tokens
    #                                     ngrams=None,  # create groups of n-words?
    #                                     output_mode="int",  # how to map tokens to numbers
    #                                     output_sequence_length=None)  # how long should the output
    #                                                                   # sequence of tokens be?
    #                                     # pad_to_max_tokens=True)  # Not valid if using max_tokens=None

    # Find average number of tokens (words) in training Tweets
    # print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))  # result = 15

    # Setup text vectorization with custom variables
    max_vocab_length = 10000  # max number of words to have in our vocabulary
    max_length = 15  # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                        output_mode='int',
                                        output_sequence_length=max_length)

    # Fit the text vectorizer to the training text
    text_vectorizer.adapt(train_sentences)

    # Create sample sentence and tokenize it
    sample_sentence = "There's a flood in my street!"
    # print(text_vectorizer([sample_sentence]))

    # Choose a random sentence from the training dataset and tokenize it
    random_sentence = random.choice(train_sentences)
    # print(f"Original text:\n{random_sentence}\
    #        \n\nVectorized version:")
    # print(text_vectorizer([random_sentence]))

    # Get the unique words in the vocabulary
    # words_in_vocab = text_vectorizer.get_vocabulary()
    # top_5_words = words_in_vocab[:5]  # most common tokens (notice the [UNK] token for "unknown" words)
    # bottom_5_words = words_in_vocab[-5:]  # least common tokens
    # print(f"Number of words in vocab: {len(words_in_vocab)}")
    # print(f"Top 5 most common words: {top_5_words}")
    # print(f"Bottom 5 least common words: {bottom_5_words}")

    '''Creating am Embedding using an Embedding Layer'''

    tf.random.set_seed(42)

    embedding = Embedding(input_dim=max_vocab_length,  # set input shape
                          output_dim=128,  # set size of embedding vector
                          embeddings_initializer='uniform',  # default, initialize randomly
                          input_length=max_length,  # how long is each input
                          name='embedding_1')

    # print(embedding)

    # Get a random sentence from training set
    random_sentence = random.choice(train_sentences)
    # print(f"Original text:\n{random_sentence}"
    #       f"\n\nEmbedded version:")

    # Embed the random sentence (turn it into numerical representation)
    sample_embed = embedding(text_vectorizer([random_sentence]))
    # print(sample_embed)

    # Check out a single token's embedding
    # print(sample_embed[0][0])
