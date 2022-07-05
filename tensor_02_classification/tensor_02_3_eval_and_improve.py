import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
import itertools

# Import function "plot_decision_boundary()" from tensor_02_1_modeling.py
from tensor_02_classification.tensor_02_1_modeling import plot_decision_boundary

# Import function for compile model from tensor_02_2_non_linearity.py
from tensor_02_classification.tensor_02_2_non_linearity import class_compile_model


def run():
    """02.3 Evaluating and improving our classification model"""

    # Getting data
    n_samples = 1000
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    # Make dataframe of features and labels
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

    # How many examples are in the whole dataset?
    # print(len(X))

    # Split data into train and test sets
    X_train, y_train = X[:800], y[:800]  # 80% of the data for the training set
    X_test, y_test = X[800:], y[800:]  # 20% of the data for the test set

    # Check the shapes of the data
    # print(X_train.shape, X_test.shape)  # 800 examples in the training set, 200 examples in the test set

    # === Build new model_8 ===
    tf.random.set_seed(42)

    model_8 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),  # hidden layer 1, using 'relu' for activation
                                                      # (same as tf.keras.activation.relu)
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # output layer, using 'sigmoid' for the output
    ])

    class_compile_model(model_8, learn_rate=0.01)  # increase learning rate from 0.001 to 0.01 for faster learning
    # history = model_8.fit(X_train, y_train, epochs=25)

    # Evaluate our model on the test set
    # loss, accuracy = model_8.evaluate(X_test, y_test)
    # print(f"Model loss on the test set: {loss}")
    # print(f"Model accuracy on the test set: {100*accuracy:.2f}%")

    # Plot the decision boundaries for the training and test sets
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Train")
    # plot_decision_boundary(model_8, X=X_train, y=y_train)
    # plt.subplot(1, 2, 2)
    # plt.title("Test")
    # plot_decision_boundary(model_8, X=X_test, y=y_test)
    # plt.show()

    '''Plot the loss curves'''

    # We can access the information in the history variable using the .history attribute
    # print(pd.DataFrame(history.history))

    # Plot the loss curves
    # pd.DataFrame(history.history).plot()
    # plt.title("Model_8 training curves")
    # plt.show()

    '''Finding the best learning rate'''

    # === Building new model_9 ===
    # tf.random.set_seed(42)
    #
    # model_9 = tf.keras.Sequential([
    #     tf.keras.layers.Dense(4, activation='relu'),
    #     tf.keras.layers.Dense(4, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid'),
    # ])
    #
    # model_9.compile(loss='binary_crossentropy',  # we can use strings here too
    #                 optimizer='Adam',  # same  as tf.keras.optimizers.Adam() with default settings
    #                 metrics=['accuracy'])

    # Create a learning rate scheduler callback
    # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

    # history = model_9.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])

    # Checkout the history
    # pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel='epochs')
    # plt.show()

    # Plot the learning rate versus the loss
    # lrs = 1e-4 * (10 ** (np.arange(100)/20))
    # plt.figure(figsize=(10, 7))
    # plt.semilogx(lrs, history.history['loss'])  # we want the x-axis (learning rate) to be log scale
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.title('Learning rate vs. loss')
    # plt.show()

    # Example of other typical learning rate values
    # print(10**0, 10**-1, 10**-2, 10**-3, 1e-4)

    # === Build new model_10 ===
    tf.random.set_seed(42)

    model_10 = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    class_compile_model(model_10, learn_rate=0.02)
    history = model_10.fit(X_train, y_train, epochs=20, verbose=0)

    # print(model_10.evaluate(X_test, y_test))

    # Plot the decision boundaries for the training and test sets
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title('Train')
    # plot_decision_boundary(model_10, X=X_train, y=y_train)
    # plt.subplot(1, 2, 2)
    # plt.title('Test')
    # plot_decision_boundary(model_10, X=X_test, y=y_test)
    # plt.show()

    '''More classification evaluation methods'''

    # Check the accuracy of our model
    loss, accuracy = model_10.evaluate(X_test, y_test)
    # print(f"Model loss on test set: {loss}")
    # print(f"Model accuracy on test set: {(accuracy*100):.2f}%")

    # Create a confusion matrix
    # Make predictions
    y_preds = model_10.predict(X_test)
    # Create confusion matrix (will be value error!)
    # confusion_matrix(y_test, y_preds)

    # View the first 10 predictions
    # print(y_preds[:10])
    # View the first 10 test labels
    # print(y_test[:10])

    # Convert prediction probabilities to binary format and view the first 10
    # print(tf.round(y_preds)[:10])

    # Create a confusion matrix
    print(confusion_matrix(y_test, tf.round(y_preds)))

    # Creating more visual confusion matrix
    # Note: The following confusion matrix code is a remix of Scikit-Learn's

    figsize = (10, 10)

    # Create the confusion matrix
    confus_matrix = confusion_matrix(y_test, tf.round(y_preds))
    confus_matrix_norm = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = confus_matrix.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(confus_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    classes = False

    if classes:
        labels = classes
    else:
        labels = np.arange(confus_matrix.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix',
           xlabel='Predicted label',
           ylabel='True label',
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colours
    threshold = (confus_matrix.max() + confus_matrix.min()) / 2.

    # Plot the test on each cell
    for i, j in itertools.product(range(confus_matrix.shape[0]), range(confus_matrix.shape[1])):
        plt.text(j, i, f"{confus_matrix[i, j]} ({confus_matrix_norm[i, j]*100:.1f}%)",
                 horizontalalignment='center',
                 color='white' if confus_matrix[i, j] > threshold else 'black',
                 size=15)

    plt.show()

    # What does itertools.product do? (Combines two things into each combination)
    for i, j in itertools.product(range(confus_matrix.shape[0]), range(confus_matrix.shape[1])):
        print(i, j)
