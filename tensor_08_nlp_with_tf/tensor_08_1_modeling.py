# Importing TensorFlow and Keras libraries
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling1D, Dense
from keras.layers import TextVectorization, Embedding
from keras.layers import LSTM, GRU, Bidirectional
from keras.optimizers import Adam
from keras import Model

# Importing Sci-kit learn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Importing other libraries
import numpy as np
import io

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
                          name='embedding_1')

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
    # model_1.summary()
    # model_1_alt.summary()

    # Fit the model
    # model_1_history = model_1.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(dir_name=TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      experiment_name="simple_dense_model")])

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
    # model_1.evaluate(val_sentences, val_labels)

    # print(embedding.weights)

    # Getting embedding weights
    embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
    # print(embed_weights.shape)

    # Make predictions (these come back in the form of probabilities)
    model_1_pred_probs = model_1.predict(val_sentences)
    # print(model_1_pred_probs[:10])  # only print out the first 10 prediction probabilities

    # Turn prediction probabilities into single-dimension tensor of floats
    model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))  # squeeze removes single dimensions
    # print(model_1_preds[:20])

    # Calculate model_1 metrics
    model_1_results = calculate_results(y_true=val_labels,
                                        y_pred=model_1_preds)
    # print(model_1_results)

    # Is our simple Keras model better than our baseline model?
    # print(np.array(list(model_1_results.values())) > np.array(list(baseline_results.values())))

    # compare_baseline_to_new_results(baseline_results=baseline_results,
    #                                 new_model_results=model_1_results)

    '''Visualizing learned embeddings'''

    # Get the vocabulary from the text vectorization layer
    words_in_vocab = text_vectorizer.get_vocabulary()
    # print(len(words_in_vocab), words_in_vocab[:10])

    # model_1.summary()

    # Get the weight matrix of embedding layer
    # (these are the numerical patterns between the text in the training dataset the model has learned)
    embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
    # print(embed_weights.shape)  # same size as vocab size and embedding_dim (each word is a embedding_dim size vector)

    # == Code below is adapted from: tensorflow.org tutorials ==

    # Create output writers
    # out_vector = io.open("training_log\\nlp_log\\embedding_vectors.tsv", "w", encoding="utf-8")
    # out_metadata = io.open("training_log\\nlp_log\\embedding_metadata.tsv", "w", encoding="utf-8")

    # # Write embedding vectors and words to file
    # for num, word in enumerate(words_in_vocab):
    #     if num == 0:
    #         continue  # skip padding token
    #     vec = embed_weights[num]
    #     out_metadata.write(word + "\n")  # write words to file
    #     out_vector.write("\t".join([str(x) for x in vec]) + "\n")  # write corresponding word vector to file
    #
    # out_vector.close()
    # out_metadata.close()

    '''Model 2: LSTM (Long-short term memory)'''

    # Set random seed and create embedding layer (new embedding layer for each model)
    tf.random.set_seed(42)
    model_2_embedding = Embedding(input_dim=max_vocab_length,
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=max_output_length,
                                  name="embedding_2")

    # === Create LSTM model ===

    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = model_2_embedding(x)
    # print(x.shape)
    # x = LSTM(64, return_sequences=True)(x)  # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
    x = LSTM(64)(x)  # return vector for whole sequence
    # print(x.shape)
    # x = Dense(64, activation="relu")(x)  # optional dense layer on top of output of LSTM cell
    outputs = Dense(1, activation="sigmoid")(x)
    model_2 = Model(inputs, outputs, name="mode_2_LSTM")

    # Compile model
    model_2.compile(loss="binary_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    # model_2.summary()

    # Fit model
    # model_2_history = model_2.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "LSTM")])

    # Make predictions on the validation dataset
    model_2_pred_probs = model_2.predict(val_sentences)
    # print(model_2_pred_probs.shape, model_2_pred_probs[:10])  # view the shape and the first 10

    # Round out predictions and reduce to 1-dimensional array
    model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
    # print(model_2_preds[:10])

    # Calculate LSTM model results
    model_2_results = calculate_results(y_true=val_labels,
                                        y_pred=model_2_preds)
    # print(model_2_results)

    # Compare model 2 to baseline
    # compare_baseline_to_new_results(baseline_results, model_2_results)

    '''Model 3: GRU'''

    # Set random seed and create embedding layer (new embedding layer` for each model)
    tf.random.set_seed(42)

    model_3_embedding = Embedding(input_dim=max_vocab_length,
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=max_output_length,
                                  name="embedding_3")

    # Build on RNN using the GRU cell
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = model_3_embedding(x)
    # x = GRU(64, return_sequences=True)  # stacking recurrent cells requires return_sequences=True
    x = GRU(64)(x)
    # x = Dense(64, activation="relu")(x)  # optional dense layer after GRU cell
    outputs = Dense(1, activation="sigmoid")(x)
    model_3 = Model(inputs, outputs, name="model_3_GRU")

    # Compile model
    model_3.compile(loss="binary_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    # Get a summary of our model look like?
    # model_3.summary()

    # Fit model
    # model_3_history = model_3.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "GRU")])

    # Make predictions on the validation data
    model_3_pred_probs = model_3.predict(val_sentences)
    # print(model_3_pred_probs.shape, model_3_pred_probs[:10])

    # Convert prediction probabilities to prediction classes
    model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
    # print(model_3_preds[:10])

    # Calculate model_3 results
    model_3_results = calculate_results(y_true=val_labels,
                                        y_pred=model_3_preds)
    # print(model_3_results)

    # Compare to baseline
    # compare_baseline_to_new_results(baseline_results, model_3_results)

    '''Model 4: Bidirectional RNN model'''

    # Set random seed and create embedding layer (new embedding layer for each model)
    tf.random.set_seed(42)

    model_4_embedding = Embedding(input_dim=max_vocab_length,
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=max_output_length,
                                  name="embedding_4")

    # Build a Bidirectional RNN in TensorFlow
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = model_4_embedding(x)
    # x = Bidirectional(LSTM(64, return_sequences=True))(x)  # stacking RNN layers requires return_sequences=True
    x = Bidirectional(LSTM(64))(x)  # bidirectional goes both ways so has double the parameters of a regular LSTM layer
    outputs = Dense(1, activation="sigmoid")(x)
    model_4 = Model(inputs, outputs, name="model_4_Bidirectional")

    # Compile model
    model_4.compile(loss="binary_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    # Get a summary
    # model_4.summary()

    # Fit the model (takes longer because of the bidirectional layers)
    # model_4_history = model_4.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "bidirectional_RNN")])

    # Make predictions with bidirectional RNN on the validation data
    model_4_pred_probs = model_4.predict(val_sentences)
    # print(model_4_pred_probs[:10])

    # Convert prediction probabilities to labels
    model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
    # print(model_4_preds[:10])

    # Calculate bidirectional RNN model results
    model_4_results = calculate_results(val_labels, model_4_preds)
    # print(model_4_results)

    # Check to see how the bidirectional model performs against the baseline
    # compare_baseline_to_new_results(baseline_results, model_4_results)

    '''Convolutional Neural Networks for Text'''




