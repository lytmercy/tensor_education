# Importing TensorFlow and Keras libraries
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, GlobalAveragePooling1D, Dense, LSTM, Conv1D, GlobalMaxPool1D
from keras.layers import TextVectorization, Embedding
from keras.optimizers import Adam
import tensorflow_hub as hub

# Importing Sci-kit learn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Importing other libraries
import numpy as np
import pandas as pd

# Importing PreprocessingData class from tensor_08_0preprocess_data.py
from tensor_08_nlp_with_tf.tensor_08_0_preprocess_data import PreprocessData

# Importing helping function
from tensor_08_nlp_with_tf.tensor_08_1_modeling import calculate_results


def run():
    """08.2 Exercises

    1. Rebuild, compile and train model_1, model_2 and model_5 using
        the Keras Sequential API instead of the Functional API.
    2. Retrain the baseline model with 10% of the training data.
        How does perform compared to the Universal Sentence Encoder model with 10% of the training data?
    3. Try fine-tuning the TF Hub Universal Sentence Encoder model
        by setting training=True when instantiating it as a Keras layer.

    We can use this encoding layer in place of our text_vectorizer and embedding layer

    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=True) # turn training on to fine-tune the TensorFlow Hub model

    4. Retrain the best model you've got so far on the whole training set (no validation split).
        Then use this trained model to make predictions on the test dataset and format the predictions
        into the same format as the sample_submission.csv file from Kaggle (see the Files tab in Colab
        for what the sample_submission.csv file looks like).
        Once you've done this, make a submission to the Kaggle competition, how did your model perform?
    5. Combine the ensemble predictions using the majority vote (mode), how does this perform compare
        to averaging the prediction probabilities of each model?
    6. Make a confusion matrix with the best performing model's predictions on the validation set
        and the validation ground truth labels.
    """

    # Initialize dataset for this file
    preprocess_data = PreprocessData()

    train_sentences, train_labels, val_sentences, val_labels = preprocess_data.get_train_val_data()

    max_vocab_length = preprocess_data.max_vocab_length
    max_output_length = preprocess_data.max_output_sequence_length

    '''Exercise - 1'''

    # Building Text Vectorization layer
    # text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
    #                                     output_mode="int",
    #                                     output_sequence_length=max_output_length)
    # Training text vectorization layer
    # text_vectorizer.adapt(train_sentences)

    # === Building Model 1 ===

    # Setting random seed for receiving expected results all time
    tf.random.set_seed(17)

    # Building Embedding layer
    # ex_embedding_1 = Embedding(input_dim=max_vocab_length,
    #                            output_dim=128,
    #                            embeddings_initializer="uniform",
    #                            input_length=max_output_length,
    #                            name="ex_embedding_1")

    # Building Sequential of model
    # ex_model_1 = Sequential([
    #     Input(shape=(1,), dtype="string"),
    #     text_vectorizer,
    #     ex_embedding_1,
    #     GlobalAveragePooling1D(),
    #     Dense(1, activation="sigmoid")
    # ])

    # Compile the model
    # ex_model_1.compile(loss="binary_crossentropy",
    #                    optimizer=Adam(),
    #                    metrics=["accuracy"])

    # Training the model
    # ex_model_1_history = ex_model_1.fit(train_sentences,
    #                                     train_labels,
    #                                     epochs=5,
    #                                     validation_data=(val_sentences, val_labels))

    # Starting print the model results
    print("|=== Model 1 Results ===|")
    # Getting prediction probabilities
    # ex_model_1_pred_probs = ex_model_1.predict(val_sentences)
    # print(ex_model_1_pred_probs[:10])

    # Convert probabilities to labels (numbers)
    # ex_model_1_preds = tf.squeeze(tf.round(ex_model_1_pred_probs))
    # print(ex_model_1_preds[:20])

    # Calculate results of predictions (accuracy, precision, recall, f1-score)
    # ex_model_1_results = calculate_results(val_labels, ex_model_1_preds)
    # print(ex_model_1_results)

    # === Building Model 2 ===
    tf.random.set_seed(17)

    # Building Embedding layer
    # ex_embedding_2 = Embedding(input_dim=max_vocab_length,
    #                            output_dim=128,
    #                            embeddings_initializer="uniform",
    #                            input_length=max_output_length,
    #                            name="ex_embedding_2")

    # Building Sequential of model
    # ex_model_2 = Sequential([
    #     Input(shape=(1,), dtype="string"),
    #     text_vectorizer,
    #     ex_embedding_2,
    #     LSTM(64),
    #     Dense(1, activation="sigmoid")
    # ])

    # Compile the model
    # ex_model_2.compile(loss="binary_crossentropy",
    #                    optimizer=Adam(),
    #                    metrics=["accuracy"])

    # Training the model
    # ex_model_2_history = ex_model_2.fit(train_sentences,
    #                                     train_labels,
    #                                     epochs=5,
    #                                     validation_data=(val_sentences, val_labels))

    # Starting print the model results
    print("|=== Model 2 Results ===|")
    # Getting prediction probabilities
    # ex_model_2_pred_probs = ex_model_2.predict(val_sentences)
    # print(ex_model_2_pred_probs[:10])

    # Convert probabilities to labels (numbers)
    # ex_model_2_preds = tf.squeeze(tf.round(ex_model_2_pred_probs))
    # print(ex_model_2_preds[:20])

    # Calculate results of predictions (accuracy, precision, recall, f1-score)
    # ex_model_2_results = calculate_results(val_labels, ex_model_2_preds)
    # print(ex_model_2_results)

    # === Building Model 5 ===
    tf.random.set_seed(17)

    # Building Embedding layer
    # ex_embedding_5 = Embedding(input_dim=max_vocab_length,
    #                            output_dim=128,
    #                            embeddings_initializer="uniform",
    #                            input_length=max_output_length,
    #                            name="ex_embedding_5")

    # Building Sequential of model
    # ex_model_5 = Sequential([
    #     Input(shape=(1,), dtype="string"),
    #     text_vectorizer,
    #     ex_embedding_5,
    #     Conv1D(filters=32, kernel_size=5, activation="relu"),
    #     GlobalMaxPool1D(),
    #     Dense(1, activation="sigmoid")
    # ])

    # Compile the model
    # ex_model_5.compile(loss="binary_crossentropy",
    #                    optimizer=Adam(),
    #                    metrics=["accuracy"])

    # Training the model
    # ex_model_5_history = ex_model_5.fit(train_sentences,
    #                                     train_labels,
    #                                     epochs=5,
    #                                     validation_data=(val_sentences, val_labels))

    # Starting print the model results
    print("|=== Model 5 Results ===|")
    # Getting prediction probabilities
    # ex_model_5_pred_probs = ex_model_5.predict(val_sentences)
    # print(ex_model_5_pred_probs[:10])

    # Convert probabilities to labels (numbers)
    # ex_model_5_preds = tf.squeeze(tf.round(ex_model_5_pred_probs))
    # print(ex_model_5_preds[:20])

    # Calculate results of predictions (accuracy, precision, recall, f1-score)
    # ex_model_5_results = calculate_results(val_labels, ex_model_5_preds)
    # print(ex_model_5_results)

    '''Exercise - 2'''

    # Create tokenization and modeling pipeline (baseline)
    # ex_model_0 = Pipeline([
    #     ("tfidf", TfidfVectorizer()),
    #     ("clf", MultinomialNB())
    # ])

    # Splitting dataset to extract 10% of the dataset
    # train_90_sentences, train_10_sentences, \
    #     train_90_labels, train_10_labels = train_test_split(np.array(train_sentences),
    #                                                         train_labels,
    #                                                         test_size=0.1,
    #                                                         random_state=17)

    # Fit the pipeline to the training data
    # ex_model_0.fit(train_10_sentences,
    #                train_10_labels)

    print("|=== Model 0 Baseline Results ===|")

    # baseline_score = ex_model_0.score(val_sentences, val_labels)
    # print(f"Out baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

    # Making predictions
    # baseline_preds = ex_model_0.predict(val_sentences)
    # print(baseline_preds[:20])

    # Get  baseline results
    # baseline_results = calculate_results(y_true=val_labels,
    #                                      y_pred=baseline_preds)

    '''Exercise - 3'''

    # ==|> Because my GPU can't load the whole Universal Sentence Encoder with trainable=True with dataset,
    # == so I used Google Colab GPU and ran the code below (file - tensor_08_2_exercise_3.ipynb)

    # # Creating tensorflow hub model encoding layer
    # sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
    #                                         input_shape=[],
    #                                         dtype=tf.string,
    #                                         trainable=True,
    #                                         name="USE")
    #
    # # Building Sequential of model
    # ex_model_use = Sequential([
    #     sentence_encoder_layer,
    #     Dense(64, activation="relu"),
    #     Dense(1, activation="sigmoid")
    # ], name="model_use")
    #
    # # Compile the model
    # ex_model_use.compile(loss="binary_crossentropy",
    #                      optimizer=Adam(),
    #                      metrics=["accuracy"])
    #
    # # Training the model
    # ex_model_use_history = ex_model_use.fit(train_sentences,
    #                                         train_labels,
    #                                         epochs=5,
    #                                         validation_data=(val_sentences, val_labels))
    #
    # # Getting prediction probabilities
    # ex_model_use_pred_probs = ex_model_use.predict(val_sentences)
    # print(ex_model_use_pred_probs[:10])
    #
    # # Convert probabilities to labels (numbers)
    # ex_model_use_preds = tf.squeeze(tf.round(ex_model_use_pred_probs))
    # print(ex_model_use_preds[:20])
    #
    # # Calculate model results (accuracy, precision, recall, f1-score)
    # ex_model_use_results = calculate_results(val_labels, ex_model_use_preds)

    '''Exercise - 4'''

