import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import random
import wget

# Import helper functions
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_histories


def run():
    """08.1 Preprocess data for NLP"""

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
    random_index = random.randint(0, len(train_df)-5)  # create random indexes not higher than the total number
    for row in train_df_shuffled[['text', 'target']][random_index:random_index+5].itertuples():
        _, text, target = row
        print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
        print(f"Text:\n{text}\n")
        print("---\n")

    '''Split data into training and validation sets'''

    # Use train_test_split to split training data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled['text'].to_numpy(),
                                                                                train_df_shuffled['target'].to_numpy(),
                                                                                test_size=0.1,  # dedicate 10% of samples to validation set
                                                                                random_state=42)  # random state for reproducibility

    # Check the lengths
    print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))

    # View the first 10 training sentences and their labels
    print(train_sentences[:10], train_labels[:10])

    '''Converting text into numbers'''

