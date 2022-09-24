# Importing TensorFlow and Keras libraries
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling1D, Dense
from keras.layers import TextVectorization, Embedding
from keras.optimizers import Adam
from keras import Model

# Importing Sci-kit learn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Importing other libraries
import numpy as np

# Importing helper functions
from helper_functions import create_tensorboard_callback

# Importing PreprocessingData class from tensor_08_0_preprocess_data.py
from tensor_08_nlp_with_tf.tensor_08_0_preprocess_data import PreprocessData


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1-score of a binary classification model;
    :param y_true: true labels in the form of a 1D array;
    :param y_pred: predicted labels in the form of a 1D array;
    :return: a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1-score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1-score": model_f1}

    return model_results


def compare_baseline_to_new_results(baseline_results, new_model_results):
    for key, value in baseline_results.items():
        print(f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, "
              f"Difference: {new_model_results[key]-value:.2f}")


def run():
    """08.1 Building NLP model and compare their"""

    # Initialize dataset for this file
    preprocess_data = PreprocessData()

    train_sentences, train_labels, val_sentences, val_labels = preprocess_data.get_train_val_data()

    '''Model 0: Getting a baseline'''

    # Create tokenization and modelling pipeline
    model_0 = Pipeline([
        ('tfidf', TfidfVectorizer()),  # convert words to numbers using tfidf
        ('clf', MultinomialNB())  # model the text
    ])

    # Fit the pipeline to the training data
    model_0.fit(train_sentences, train_labels)

    # baseline_score = model_0.score(val_sentences, val_labels)
    # print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")

    # Make predictions
    baseline_preds = model_0.predict(val_sentences)
    # print(baseline_preds[:20])

    '''Creating an evaluation function for our model experiments'''

    # === Creating function calculate_results for binary classification above ^ ===

    # Get baseline results
    baseline_results = calculate_results(y_true=val_labels,
                                         y_pred=baseline_preds)
    # print(baseline_results)

    '''Model 1: A simple dense model'''

    # Create directory to save TensorBoard logs
    TENSOR_CALLBACK_SAVE_DIR = "training_log\\nlp_log"

    max_vocab_length = preprocess_data.max_vocab_length
    max_output_length = preprocess_data.max_output_sequence_length

    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                        output_mode='int',
                                        output_sequence_length=max_output_length)

    text_vectorizer.adapt(train_sentences)

    tf.random.set_seed(42)

    embedding = Embedding(input_dim=max_vocab_length,
                          output_dim=128,
                          embeddings_initializer='uniform',
                          input_length=max_output_length,
                          name='embedding_2')

    inputs = Input(shape=(1,), dtype="string")  # inputs are 1-dimensional strings
    x = text_vectorizer(inputs)  # turn the input text into numbers
    x = embedding(x)  # create an embedding of the numerized numbers
    x = GlobalAveragePooling1D()(x)  # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
    outputs = Dense(1, activation="sigmoid")(x)  # create the output layer, want binary outputs so use sigmoid activation
    model_1 = Model(inputs, outputs, name="model_1_dense")  # construct the model

    # Exercise - 1
    # Try building model_1 with and without a GlobalAveragePooling1D() layer after the embedding layer. What happens?
    # Why do you think this is?

    x_1 = text_vectorizer(inputs)
    x_1 = embedding(x_1)
    outputs_1 = Dense(1, activation="sigmoid")(x_1)
    model_1_alt = Model(inputs, outputs_1, name="model_1_alternate")

    # Compile model
    model_1.compile(loss="binary_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    model_1_alt.compile(loss="binary_crossentropy",
                        optimizer=Adam(),
                        metrics=["accuracy"])

    # Get a summary of the model
    model_1.summary()
    model_1_alt.summary()

    # Fit the model
    model_1_history = model_1.fit(train_sentences,
                                  train_labels,
                                  epochs=5,
                                  validation_data=(val_sentences, val_labels),
                                  callbacks=[create_tensorboard_callback(dir_name=TENSOR_CALLBACK_SAVE_DIR,
                                                                         experiment_name="simple_dense_model")])

    # === Exercise -1
    #   === Commented because model without a GlobalAveragePooling1D() layer after the embedding layer, gives an error
    #   === ValueError: `logits` and `labels` must have the same shape, received ((None, 15, 1) vs (None,)).
    # model_1_alt_history = model_1_alt.fit(train_sentences,
    #                                       train_labels,
    #                                       epochs=5,
    #                                       validation_data=(val_sentences, val_labels),
    #                                       callbacks=[create_tensorboard_callback(dir_name=TENSOR_CALLBACK_SAVE_DIR,
    #                                                                              experiment_name="simple_alt_dense_model")])

    # Check the results
    model_1.evaluate(val_sentences, val_labels)

    print(embedding.weights)

    # Getting embedding weights
    embed_weights = model_1.get_layer("embedding_2").get_weights()[0]
    print(embed_weights.shape)

    # Make predictions (these come back in the form of probabilities)
    model_1_pred_probs = model_1.predict(val_sentences)
    print(model_1_pred_probs[:10])  # only print out the first 10 prediction probabilities

    # Turn prediction probabilities into single-dimension tensor of floats
    model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))  # squeeze removes single dimensions
    print(model_1_preds[:20])

    # Calculate model_1 metrics
    model_1_results = calculate_results(y_true=val_labels,
                                        y_pred=model_1_preds)
    print(model_1_results)

    # Is our simple Keras model better than our baseline model?
    print(np.array(list(model_1_results.values())) > np.array(list(baseline_results.values())))

    compare_baseline_to_new_results(baseline_results=baseline_results,
                                    new_model_results=model_1_results)

    '''Visualizing learned embeddings'''


