# Importing TensorFlow and Keras libraries
import tensorflow as tf
import tensorflow_hub as hub
from keras import Sequential
from keras.layers import Input, GlobalAveragePooling1D, Dense
from keras.layers import TextVectorization, Embedding
from keras.layers import LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPool1D
from keras.optimizers import Adam
from keras.models import clone_model
from keras import Model
from keras.models import load_model

# Importing Sci-kit learn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Importing other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import random
import time

# Importing helper functions
from helper_functions import create_tensorboard_callback

# Importing PreprocessingData class from tensor_08_0_preprocess_data.py
from tensor_08_nlp_with_tf.tensor_08_0_preprocess_data import PreprocessData

# Create directory to save TensorBoard logs
TENSOR_CALLBACK_SAVE_DIR = "training_log\\nlp_log"


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


def predict_on_sentence(model, sentence):
    """
    Uses model to make a prediction on sentence;
    :param model: pretrained model;
    :param sentence: sample of sentence for prediction.
    :return: the sentence, the predicted label and the prediction probability.
    """
    pred_prob = model.predict([sentence])
    pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
    print(f"Pred: {pred_label}", "(real disaster)" if pred_label > 0 else "(not real disaster)", f"Prob: {pred_prob[0][0]}")
    print(f"Text:\n{sentence}")


def pred_timer(model, samples):
    """
    Times how long a model takes to make predictions on samples;
    :param model: pretrained model;
    :param samples: a list of text samples;
    :return: total_time = total elapsed time for model to make predictions on samples;
    time_per_pred = time in seconds per single sample.
    """
    start_time = time.perf_counter()  # get start time
    model.predict(samples)  # make predictions
    end_time = time.perf_counter()  # get finish time
    total_time = end_time - start_time  # calculate how long predictions took to make
    time_per_pred = total_time/len(samples)  # find prediction time per sample
    return total_time, time_per_pred


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

    # x_1 = text_vectorizer(inputs)
    # x_1 = embedding(x_1)
    # outputs_1 = Dense(1, activation="sigmoid")(x_1)
    # model_1_alt = Model(inputs, outputs_1, name="model_1_alternate")

    # Compile model
    # model_1.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

    # model_1_alt.compile(loss="binary_crossentropy",
    #                     optimizer=Adam(),
    #                     metrics=["accuracy"])

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
    # embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
    # print(embed_weights.shape)

    # Make predictions (these come back in the form of probabilities)
    # model_1_pred_probs = model_1.predict(val_sentences)
    # print(model_1_pred_probs[:10])  # only print out the first 10 prediction probabilities

    # Turn prediction probabilities into single-dimension tensor of floats
    # model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))  # squeeze removes single dimensions
    # print(model_1_preds[:20])

    # Calculate model_1 metrics
    # model_1_results = calculate_results(y_true=val_labels,
    #                                     y_pred=model_1_preds)
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
    # embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
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

    '''Recurrent Neural Networks (RNN)'''
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
    # model_2.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

    # model_2.summary()

    # Fit model
    # model_2_history = model_2.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "LSTM")])

    # Make predictions on the validation dataset
    # model_2_pred_probs = model_2.predict(val_sentences)
    # print(model_2_pred_probs.shape, model_2_pred_probs[:10])  # view the shape and the first 10

    # Round out predictions and reduce to 1-dimensional array
    # model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
    # print(model_2_preds[:10])

    # Calculate LSTM model results
    # model_2_results = calculate_results(y_true=val_labels,
    #                                     y_pred=model_2_preds)
    # print(model_2_results)

    # Compare model 2 to baseline
    # compare_baseline_to_new_results(baseline_results, model_2_results)

    '''Model 3: GRU (Gated Recurrent Unit)'''

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
    # model_3.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

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
    # model_3_pred_probs = model_3.predict(val_sentences)
    # print(model_3_pred_probs.shape, model_3_pred_probs[:10])

    # Convert prediction probabilities to prediction classes
    # model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
    # print(model_3_preds[:10])

    # Calculate model_3 results
    # model_3_results = calculate_results(y_true=val_labels,
    #                                     y_pred=model_3_preds)
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
    # model_4.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

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
    # model_4_pred_probs = model_4.predict(val_sentences)
    # print(model_4_pred_probs[:10])

    # Convert prediction probabilities to labels
    # model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
    # print(model_4_preds[:10])

    # Calculate bidirectional RNN model results
    # model_4_results = calculate_results(val_labels, model_4_preds)
    # print(model_4_results)

    # Check to see how the bidirectional model performs against the baseline
    # compare_baseline_to_new_results(baseline_results, model_4_results)

    '''Convolutional Neural Networks for Text'''
    '''Model 5: Conv1D'''

    # Test out the embedding, 1D convolutional and max pooling
    embedding_test = embedding(text_vectorizer(["this is a test sentence"]))  # turn target sentence into embedding
    conv_1d = Conv1D(filters=32, kernel_size=5, activation="relu")  # convolve over target sequence 5 words at a time
    conv_1d_output = conv_1d(embedding_test)  # pass embedding through 1D convolutional layer
    max_pool = GlobalMaxPool1D()
    max_pool_output = max_pool(conv_1d_output)  # get the most important features
    # print(embedding_test.shape, conv_1d_output.shape, max_pool_output.shape)

    # See the outputs of each layer
    # print(embedding_test[:1], conv_1d_output[:1], max_pool_output[:1])

    # Set random seed and create embedding layer (new embedding layer for each model)
    tf.random.set_seed(42)
    model_5_embedding = Embedding(input_dim=max_vocab_length,
                                  output_dim=128,
                                  embeddings_initializer="uniform",
                                  input_length=max_output_length,
                                  name="embedding_5")

    # Create 1-dimensional convolutional layer to model sequences
    inputs = Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = model_5_embedding(x)
    x = Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = GlobalMaxPool1D()(x)
    # x = Dense(64, activation="relu")(x)  # optional dense layer
    outputs = Dense(1, activation="sigmoid")(x)
    model_5 = Model(inputs, outputs, name="model_5_Conv1D")

    # Compile model
    # model_5.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

    # Get a summary of our 1D Convolution model
    # model_5.summary()

    # Fit the model
    # model_5_history = model_5.fit(train_sentences,
    #                               train_labels,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "Conv1D")])

    # Make predictions with model_5
    # model_5_pred_probs = model_5.predict(val_sentences)
    # print(model_5_pred_probs[:10])

    # Convert model_5 prediction probabilities to labels
    # model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
    # print(model_5_preds[:10])

    # Calculate model_5 evaluation metrics
    # model_5_results = calculate_results(y_true=val_labels,
    #                                     y_pred=model_5_preds)
    # print(model_5_results)

    # Compare model_5 results to baseline
    # compare_baseline_to_new_results(baseline_results, model_5_results)

    '''Using Pretrained Embeddings (transfer learning for NLP)'''
    '''Model 6: TensorFlow Hub Pretrained sentence Encoder'''

    sample_sentence = "There's a flood in my street!"
    # Example of pretrained embedding with universal sentence encoder
    # embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")  # load Universal Sentence Encoder
    # embed_samples = embed([sample_sentence,
    #                        "When you call the unisersal sentence encode on a sentence, it turns it into numbers."])
    # print(embed_samples[0][:50])

    # Each sentence has been encoded into a 512 dimension vector
    # print(embed_samples[0].shape)

    # We can use this encoding layer in place of our text_vectorizer and embedding layer
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            input_shape=[],  # shape of inputs coming to our model
                                            dtype=tf.string,  # data type of input coming to the USE layer
                                            trainable=False,  # keep the pretrained weight
                                                              # (we'll create a feature extractor)
                                            name="USE")

    # Create model using the Sequential API
    model_6 = Sequential([
        sentence_encoder_layer,
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ], name="model_6_USE")

    # Compile model
    model_6.compile(loss="binary_crossentropy",
                    optimizer=Adam(),
                    metrics=["accuracy"])

    # model_6.summary()

    # Train a classifier on top of pretrained embeddings
    model_6_history = model_6.fit(train_sentences,
                                  train_labels,
                                  epochs=5,
                                  validation_data=(val_sentences, val_labels),
                                  callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
                                                                         "tf_hub_sentence_encoder")])

    # Make predictions with USE TF Hub model
    model_6_pred_probs = model_6.predict(val_sentences)
    # print(model_6_pred_probs[:10])

    # Convert prediction probabilities to labels
    model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
    # print(model_6_preds[:10])

    # Calculate model 6 performance metrics
    model_6_results = calculate_results(val_labels, model_6_preds)
    # print(model_6_results)

    # Compare TF Hub model to baseline
    # compare_baseline_to_new_results(baseline_results, model_6_results)

    '''Model 7: TensorFlow Hub Pretrained Sentence Encoder 10% of the training data'''

    # NOTE: Making splits like this will lead to data leakage
    # (some training examples in the validation set)

    # === Wrong way to make splits (train_df_shuffled has already been split) ===

    # One kind of correct way (there are more) to make data subset
    # (split the already split  train_sentences/train_labels)
    train_sentences_90_percent, train_sentences_10_percent, \
    train_labels_90_percent, train_labels_10_percent = train_test_split(np.array(train_sentences),
                                                                        train_labels,
                                                                        test_size=0.1,
                                                                        random_state=42)

    # Check Length of 10 percent datasets
    # print(f"Total training examples: {len(train_sentences)}")
    # print(f"Length of 10% training examples: {len(train_sentences_10_percent)}")

    # Check the number of targets in our subset of data
    # (this should be cose to the distribution of labels in the original train_labels)
    # print(pd.Series(train_labels_10_percent).value_counts())

    # Clone model_6 but reset weights
    # model_7 = clone_model(model_6)

    # Creating pretrained embedding from scratch because my GPU can't load more than one pretrained embedding.
    # sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
    #                                         input_shape=[],  # shape of inputs coming to our model
    #                                         dtype=tf.string,  # data type of input coming to the USE layer
    #                                         trainable=False,  # keep the pretrained weight
    #                                                           # (we'll create a feature extractor)
    #                                         name="USE")

    # Create model using the Sequential API
    # model_7 = Sequential([
    #     sentence_encoder_layer,
    #     Dense(64, activation="relu"),
    #     Dense(1, activation="sigmoid")
    # ], name="model_6_USE")

    # Compile model
    # model_7.compile(loss="binary_crossentropy",
    #                 optimizer=Adam(),
    #                 metrics=["accuracy"])

    # Get a summary (will be some as model_6)
    # model_7.summary()

    # Fit the model to 10% of the training data
    # model_7_history = model_7.fit(x=train_sentences_10_percent,
    #                               y=train_labels_10_percent,
    #                               epochs=5,
    #                               validation_data=(val_sentences, val_labels),
    #                               callbacks=[create_tensorboard_callback(TENSOR_CALLBACK_SAVE_DIR,
    #                                                                      "10prcnt_tf_hub_sentence_encoder")])

    # Make predictions with the model trained on 10% of the data
    # model_7_pred_probs = model_7.predict(val_sentences)
    # print(model_7_pred_probs[:10])

    # Convert prediction probabilities to labels
    # model_7_preds = tf.squeeze(tf.round(model_7_pred_probs))
    # print(model_7_preds[:10])

    # Calculate model results
    # model_7_results = calculate_results(val_labels, model_7_preds)
    # print(model_7_results)

    # Compare to baseline
    # compare_baseline_to_new_results(baseline_results, model_7_results)

    '''Comparing the performance of each of our models'''

    # Combine model results into a DataFrame
    # all_model_results = pd.DataFrame({"baseline": baseline_results,
    #                                   "simple_dense": model_1_results,
    #                                   "lstm": model_2_results,
    #                                   "gru": model_3_results,
    #                                   "bidirectional": model_4_results,
    #                                   "conv1d": model_5_results,
    #                                   "tf_hub_sentence_encoder": model_6_results})

    # Another all_model_results with model_7 instead model_6
    # all_model_results = pd.DataFrame({"baseline": baseline_results,
    #                                   "simple_dense": model_1_results,
    #                                   "lstm": model_2_results,
    #                                   "gru": model_3_results,
    #                                   "bidirectional": model_4_results,
    #                                   "conv1d": model_5_results,
    #                                   "tf_hub_10_sentence_encoder": model_7_results})

    # all_model_results = all_model_results.transpose()
    # print(all_model_results)

    # Reduce the accuracy to same scale as other metrics
    # all_model_results["accuracy"] = all_model_results["accuracy"]/100

    # Plot and compare all the model results
    # all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0))
    # plt.show()

    # Sort model results by f1-score
    # all_model_results.sort_values("f1-score", ascending=False)["f1-score"].plot(kind="bar", figsize=(10, 7))
    # plt.show()

    # === View tensorboard logs of transfer learning modelling experiments (should be 4 models)
    # = Upload TensorBoard dev records
    # tensorboard dev upload( --logdir ./training_log/nlp_log \
    #   --name "NLP modelling experiments" \
    #   --description "A series of different NLP modellings experiments with various models" \
    #   --one_shot  # exits the uploader when upload has finished

    '''Combining our models (model ensembling/stacking)'''

    # Get mean pred probs for 3 models
    # baseline_pred_probs = np.max(model_0.predict_proba(val_sentences), axis=1)  # get the prediciton probabilities from baseline model
    # combined_pred_probs = baseline_pred_probs + tf.squeeze(model_2_pred_probs, axis=1) + tf.squeeze(model_6_pred_probs)
    # combined_preds = tf.round(combined_pred_probs/3)  # average and round the prediction probabilities to get prediction classes
    # print(combined_preds[:20])

    # Calculate results from averaging the prediction probabilities
    # ensemble_results = calculate_results(val_labels, combined_preds)
    # print(ensemble_results)

    # Add our combined model's results to the results DataFrame
    # all_model_results.loc["ensemble_results"] = ensemble_results

    # Convert the accuracy to the same scale as the rest of the results
    # all_model_results["accuracy"] = all_model_results["accuracy"]/100

    # print(all_model_results)

    '''Saving and loading a trained model'''

    # Save TF Hub Sentence Encoder model to HDF5 format
    # model_6.save("models\\nlp_model\\model_6.h5")

    # Load model with custom Hub Layer (required with HDF5 format)
    # loaded_model_6 = load_model("models\\nlp_model\\model_6.h5",
    #                             custom_objects={"KerasLayer": hub.KerasLayer})

    # How does our loaded model perform?
    # print(loaded_model_6.evaluate(val_sentences, val_labels))

    # Save TF Hub Sentence Encoder model to SavedModel format (default)
    # model_6.save("models\\nlp_model\\model_6_SavedModel_format")

    # Load TF Hub Sentence Encoder SavedModel
    # loaded_model_6_SavedModel = load_model("models\\nlp_model\\model_6_SavedModel_format")

    # Evaluate loaded SavedModel format
    # print(loaded_model_6_SavedModel.evaluate(val_sentences, val_labels))

    '''Finding the most wrong examples'''

    # Create dataframe with validation sentences and best performing model predictions
    # val_df = pd.DataFrame({"text": val_sentences,
    #                        "target": val_labels,
    #                        "pred": model_6_preds,
    #                        "pred_prob": tf.squeeze(model_6_pred_probs)})

    # print(val_df.head())

    # Find the wrong predictions and sort by prediction probabilities
    # most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)
    # print(most_wrong[:10])

    # Check the false positives (model predicted 1 when should've been 0)
    # for row in most_wrong[:10].itertuples():  # loop through the top 10 rows (change the index to view different rows)
    #     _, text, target, pred, prob = row
    #     print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
    #     print(f"Text:\n{text}\n")
    #     print("----\n")

    # Check the most wrong false negatives (model predicted 0 when should've predicted 1)
    # for row in most_wrong[-10:].itertuples():
    #     _, text, target, pred, prob = row
    #     print(f"Target: {target}, Pred: {int(pred)}, Prob: {prob}")
    #     print(f"Text:\n{text}\n")
    #     print("----\n")

    '''Making predictions on the test dataset'''

    # Define test dataframe
    # test_df = preprocess_data.get_test_dataframe()
    # Making predictions
    # test_sentences = test_df["text"].to_list()
    # test_samples = random.sample(test_sentences, 10)
    # for test_sample in test_samples:
    #     pred_prob = tf.squeeze(model_6.predict([test_sample]))  # has to be list
    #     pred = tf.round(pred_prob)
    #     print(f"Pred: {int(pred)}, Prob: {pred_prob}")
    #     print(f"Text:\n{test_sample}n")
    #     print("----\n")

    '''Predicting on Tweets from the wild'''

    # Turn Tweet into string
    some_tweet = "Life like an ensemble: take the best choices from others and make your own"

    # === Creating predict_on_sentence function for taking model & sample_sentence and return a prediction ===

    # Make a prediction on Tweet from the wild
    # predict_on_sentence(model=model_6,  # use the USE model
    #                     sentence=some_tweet)

    # Source from Tweeter
    beirut_tweet_1 = "Reports that the smoke in Beirut sky contains nitric acid, which is toxic. Please share and refrain from stepping outside unless urgent. #Lebanon"

    # Source from Tweeter
    beirut_tweet_2 = "#Beirut declared a “devastated city”, two-week state of emergency officially declared. #Lebanon"

    # Predict on disaster Tweet 1
    # predict_on_sentence(model=model_6,
    #                     sentence=beirut_tweet_1)

    # Predict on disaster Tweet 2
    # predict_on_sentence(model=model_6,
    #                     sentence=beirut_tweet_2)

    '''The speed/score tradeoff'''

    # === Creating function for calculate the time of predictions ===

    # Calculate TF Hub Sentence Encoder prediction times
    model_6_total_pred_time, model_6_time_per_pred = pred_timer(model_6, val_sentences)
    print(model_6_total_pred_time, model_6_time_per_pred)

    # Calculate Naive Bayes prediction times
    baseline_total_pred_time, baseline_time_per_pred = pred_timer(model_0, val_sentences)
    print(baseline_total_pred_time, baseline_time_per_pred)

    # Let's compare time per prediction versus our model's F1-scores
    plt.figure(figsize=(10, 7))
    plt.scatter(baseline_time_per_pred, baseline_results["f1-score"], label="baseline")
    plt.scatter(model_6_time_per_pred, model_6_results["f1-score"], label="tf_hub_sentence_encoder")
    plt.legend()
    plt.title("F1-score versus time per prediction")
    plt.xlabel("Time per prediction")
    plt.ylabel("F1-score")
    plt.show()
