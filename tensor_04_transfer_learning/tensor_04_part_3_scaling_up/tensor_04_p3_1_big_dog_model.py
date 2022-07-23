import tensorflow as tf
from keras.models import Sequential, load_model
from keras import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight, Rescaling
from keras.optimizers import Adam
from keras.applications import EfficientNetB0
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import random
import os
import wget

# Import helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories, unzip_data

# Import class for get train and test data
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_0_beginning import GettingData


def create_model(num_classes):
    # === Build dataaugmentation layer ===
    data_augmentation = Sequential([
        RandomFlip('horizontal'),  # randomly flip images on horizontal edge
        RandomRotation(0.2),  # randomly rotate images by a specific amount
        RandomHeight(0.2),  # randomly adjust the height of an image by a specific  amount
        RandomWidth(0.2),  # randomly adjust the width of an image by a specific amount
        RandomZoom(0.2),  # randomly zoom into an image
        # Rescaling(1./255)  # keep for models like ResNet50V2, remove for EfficientNet
    ], name='data_augmentation')

    # === Build model ===
    # Setup base model and freeze its layers (this will extract features)
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Setup model architecture with trainable top layers
    inputs = Input(shape=(224, 224, 3), name='input_layer')  # shape of input image
    x = data_augmentation(inputs)  # augment images (only happens during training)
    x = base_model(x, training=False)  # put the base model in inference mode, so we can use it to
                                       # extract features without updating the weights
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)  # pool the outputs of the base model
    # same number of outputs as classes
    outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)
    model = Model(inputs, outputs)

    return model, base_model


def compile_model(model, learn_rate=0.001):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learn_rate),
                  metrics=['accuracy'])


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values will be used.

    Example usage:
        make_confusion_matrix(y_true=test_labels,  # grond truth test labels
                              y_pred=y_preds,  # predicted labels
                              classes=class_names,  # array of class label names
                              figsize=(15, 15),
                              text_size=10)

    :param y_true: Array of truth labels (must be same shape as y_pred).
    :param y_pred: Array of predicted labels (must be same shape as y_true).
    :param classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    :param figsize: Size of output figure (default=(10, 10)).
    :param text_size: Size of output figure text (default=15).
    :param norm: normalize values or not (default=False).
    :param savefig: save confusion matrix to file (default=False).

    :return: A labelled confusion matrix plot comparing y_true and y_pred.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colours will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # ## Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set the threshold for different colours
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%",
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black',
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment='center',
                     color='white' if cm[i, j] > threshold else 'black',
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig('confusion_matrix.png')


def autolabel(rects, ax):
    """Attach a text label above each bar displaying its height (it's value)."""
    for rect in rects:
        width = rect.get_width()
        ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
                f"{width:.2f}",
                ha='center', va='bottom')


def load_and_prep_image(filename, img_shape=224, scale=True):
    """Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    :param filename: (str) string filename of target image
    :param img_shape: (int) size to resize target image to, default 224
    :param scale: bool whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img/255.
    else:
        return img


def make_pred_and_visualize(model, food_datasets, class_names):
    for img in food_datasets:
        img = load_and_prep_image(img, scale=False)
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]
        # Plot the images with appropriate annotations
        plt.figure()
        plt.imshow(img/255.)
        plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
        plt.axis(False)
        plt.show()


def run():
    """04.p3.1 Big dog model with transfer learning"""

    '''Train a big dog model with transfer learning on 10% of 101 food classes'''

    # Setup data inputs
    get_data_class = GettingData()

    train_data_all_10_percent = get_data_class.get_train_data()
    test_data = get_data_class.get_test_data()

    # Creating checkpoint callback to save model for later use
    checkpoint_path = 'checkpoints\\101_classes_10_percent_data_model_checkpoint\\checkpoint.ckpt'
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          # save only the model weights
                                          save_weights_only=True,
                                          # save the model weights which score the best validation accuracy
                                          monitor='val_accuracy',
                                          # only keep the best model weights on file (delete the rest)
                                          save_best_only=True)

    # === Create function for building model above ===

    # Use function
    model, base_model = create_model(len(train_data_all_10_percent.class_names))

    # model.summary()

    # Compile
    compile_model(model)

    # Fit
    # history_all_classes_10_percent = model.fit(train_data_all_10_percent,
    #                                            epochs=5,  # fit for 5 epochs to keep experiments quick
    #                                            validation_data=test_data,
    #                                            # evaluate on smaller portion of test data
    #                                            validation_steps=int(0.15 * len(test_data)),
    #                                            # save best model weights to file
    #                                            callbacks=[checkpoint_callback])

    # Evaluate model
    # results_feature_extraction_model = model.evaluate(test_data)
    # print(results_feature_extraction_model)

    # plot_loss_curves(history_all_classes_10_percent)
    # plt.show()

    '''Fine-tuning'''

    # Unfreeze all the layers in the base model
    base_model.trainable = True

    # Refreeze every layer except for the last 5
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    # Recompile model with lower learning rate
    compile_model(model, 1e-4)  # 10x lower learning rate than default

    # What layers in the model are trainable?
    # for layer in model.layers:
    #     print(layer.name, layer.trainable)

    # Check which layers are trainable
    # for layer_number, layer in enumerate(base_model.layers):
    #     print(layer_number, layer.name, layer.trainable)

    # Fine-tune for 5 more epochs
    fine_tune_epochs = 10  # model has already done 5 epochs, this is the total number of epochs we're after (5+5=10)

    # history_all_classes_10_percent_fine_tune = model.fit(train_data_all_10_percent,
    #                                                      epochs=fine_tune_epochs,
    #                                                      validation_data=test_data,
    #                                                      # validate on 15% of the test data
    #                                                      validation_steps=int(.15 * len(test_data)),
    #                                                      # start from previous last epoch
    #                                                      initial_epoch=history_all_classes_10_percent.epoch[-1])

    # Evaluate fine-tune model on the whole test dataset
    # results_all_classes_10_percent_fine_tune = model.evaluate(test_data)
    # print(results_all_classes_10_percent_fine_tune)  # results: [1.4919276237487793, 0.6063366532325745]

    # compare_histories(original_history=history_all_classes_10_percent,
    #                   new_history=history_all_classes_10_percent_fine_tune,
    #                   initial_epochs=5)

    '''Saving our trained model'''

    # # Save model to drive, so it can be used later
    # model.save('models\\101_food_class_10_percent_saved_big_dog_model')

    '''Evaluating the performance of the big dog model across all different classes'''

    # Note: loading a model will output a lot of 'WARGNINGS', these can be ignored:
    # https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training
    model = load_model('models\\101_food_class_10_percent_saved_big_dog_model')

    # Check to see if loaded model is a trained model
    # loaded_loss, loaded_accuracy = model.evaluate(test_data)
    # print(loaded_loss, loaded_accuracy)

    '''Making predictions with our trained model'''

    # Make predictions with model
    pred_probs = model.predict(test_data, verbose=1)  # set verbosity to see how long it will take

    # How many predictions are there?
    # print(len(pred_probs))

    # What's the shape of our predictions?
    # print(pred_probs.shape)

    # How do they look?
    # print(pred_probs[:10])

    # We get one prediction probability per class
    # print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
    # print(f"What prediction probability sample 0 looks like:\n{pred_probs[0]}")
    # print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")

    # Get the class predictions of each label
    pred_classes = pred_probs.argmax(axis=1)

    # How do they look?
    # print(pred_classes[:10])

    # Note: This might take a minute or so due to unravelling 790 batches
    y_labels = []
    for images, labels in test_data.unbatch():  # unbatch the test data and get images and labels
        y_labels.append(labels.numpy().argmax())  # append the index which has the largest value (labels are one-hot)
    # print(y_labels[:10])  # check what they look like (unshuffled)

    # How many labels are there? (should be the same as how many prediction probabilities we have)
    # print(len(y_labels))

    '''Evaluating our models predictions'''

    # Get accuracy score by comparing predicted classes to ground truth labels
    # sklearn_accuracy = accuracy_score(y_labels, pred_classes)
    # print(sklearn_accuracy)

    # Does the evaluate method compare to te Scikit-Learn measured accuracy?
    # print(f"Close? : {np.isclose(loaded_accuracy, sklearn_accuracy)} | "
    #       f"Difference: {loaded_accuracy - sklearn_accuracy}")

    # === Create new function for make confusion matrix for 101 classes remix above ^ ===

    # Use Function
    class_names = test_data.class_names
    # print(class_names)

    # Plot a confusion matrix with all 25250 predictions, ground truth labels and 101 classes
    # make_confusion_matrix(y_true=y_labels,
    #                       y_pred=pred_classes,
    #                       classes=class_names,
    #                       figsize=(100, 100),
    #                       text_size=20,
    #                       norm=False,
    #                       savefig=True)

    # print(classification_report(y_labels, pred_classes))

    # Get a dictionary of the classification report
    classification_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
    # print(classification_report_dict)

    # Create empty dictionary
    class_f1_scores = {}
    # Loop through classification report items
    for k, v in classification_report_dict.items():
        if k == 'accuracy':  # stop once we get to accuracy key
            break
        else:
            # Append class names and f1-scores to new dictionary
            class_f1_scores[class_names[int(k)]] = v['f1-score']
    # print(class_f1_scores)

    # Turn f1-scores into dataframe for visualization
    f1_scores = pd.DataFrame({'class_name': list(class_f1_scores.keys()),
                              'f1-score': list(class_f1_scores.values())}).sort_values('f1-score', ascending=False)
    # print(f1_scores)

    # === Create function which is modified version of matplotlib/barchart_demo above ===

    # Use function
    # fig, ax = plt.subplots(figsize=(12, 25))
    # scores = ax.barh(range(len(f1_scores)), f1_scores['f1-score'].values)
    # ax.set_yticks(range(len(f1_scores)))
    # ax.set_yticklabels(list(f1_scores['class_name']))
    # ax.set_xlabel('f1-score')
    # ax.set_title("F1-Scores for 10 Different Classes")
    # ax. invert_yaxis()  # reverse the order

    # autolabel(scores, ax)
    # plt.show()

    # Exercise: Visualize some of the most poor performing classes.
    # 8   bread_pudding  0.328358
    # 93          steak  0.323770
    # 82        ravioli  0.307692
    # 39      foie_gras  0.214286
    # 0       apple_pie  0.212121

    # i = 0
    # plt.figure(figsize=(12, 9))
    # for k in class_f1_scores:
    #     if class_f1_scores[k] < 0.35:
    #         class_name = k
    #         filename = random.choice(os.listdir(get_data_class.test_dir + class_name))
    #         filepath = get_data_class.test_dir + class_name + '\\' + filename
    #
    #         img = load_and_prep_image(filepath, scale=False)
    #         plt.subplot(2, 3, i+1)
    #         plt.imshow(img/255.)
    #         plt.title(class_name)
    #         plt.axis(False)
    #         i += 1
    # plt.show()

    '''Visualizing predictions on test images'''

    # plt.figure(figsize=(10, 7))
    # for i in range(4):
    #     # Choose a random image from a random class
    #     class_name = random.choice(class_names)
    #     filename = random.choice(os.listdir(get_data_class.test_dir + class_name))
    #     filepath = get_data_class.test_dir + class_name + '\\' + filename
    #
    #     # Load the image and make predictions
    #     img = load_and_prep_image(filepath, scale=False)  # don't scale images for EfficientNet predictions
    #     pred_prob = model.predict(tf.expand_dims(img, axis=0))  # model accepts tensors of shape [None, 224, 224, 3]
    #     pred_class = class_names[pred_prob.argmax()]  # find the predicted class
    #
    #     # Plot the images(s)
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(img/255.)
    #     if class_name == pred_class:
    #         title_color = 'g'
    #     else:
    #         title_color = 'r'
    #
    #     plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
    #     plt.axis(False)
    # plt.show()

    '''Finding the most wrong predictions'''

    # 1. Get the filenames of all our test data
    filepaths = []
    for filepath in test_data.list_files('datasets\\101_food_classes_10_percent\\test\\*\\*.jpg',
                                         shuffle=False):
        filepaths.append(filepath.numpy())

    # print(filepaths[:10])

    # 2. Create a dataframe out of current prediction data for analysis
    pred_df = pd.DataFrame({'img_path': filepaths,
                            'y_true': y_labels,
                            'y_pred': pred_classes,
                            'pred_conf': pred_probs.max(axis=1),  # get the maximum prediction probability value
                            'y_true_classname': [class_names[i] for i in y_labels],
                            'y_pred_classname': [class_names[i] for i in pred_classes]})
    print(pred_df.head())

    # 3. Is the prediction correct?
    pred_df['pred_correct'] = pred_df['y_true'] == pred_df['y_pred']
    # print(pred_df.head())

    # 4. Get the top 100 wrong examples
    top_100_wrong = pred_df[pred_df['pred_correct'] == False].sort_values('pred_conf', ascending=False)[:100]
    print(top_100_wrong.head(20))

    # 5. Visualize some of the most wrong examples
    images_to_view = 9
    start_index = 10  # change the start index to view more
    plt.figure(figsize=(15, 10))
    for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
        plt.subplot(3, 3, i+1)
        img = load_and_prep_image(row[1], scale=True)
        _, _, _, _, pred_prob, y_true, y_pred, _ = row  # only interested in a few parameters of each row
        plt.imshow(img)
        plt.title(f"actual: {y_true}, pred: {y_pred} \nprob: {pred_prob:.2f}")
        plt.axis(False)

    plt.show()

    '''Test out the big dog model on test images as well as custom images of food'''

    # Download some custom images from Google Storage
    # wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/custom_food_images.zip')
    # unzip_data('custom_food_images.zip')

    custom_food_images = ['datasets\\custom_food_images\\' + img_path
                          for img_path in os.listdir('datasets\\custom_food_images')]
    # print(custom_food_images)

    # Make predictions on custom food images
    make_pred_and_visualize(model, custom_food_images, class_names)
