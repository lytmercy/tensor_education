import tensorflow as tf
from keras.models import Sequential, load_model
from keras import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight, Rescaling
from keras.optimizers import Adam
from keras.applications import EfficientNetB0
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_histories

# Import class for get train and test data
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_0_beginning import GettingData


def compile_model(model, learn_rate=0.001):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=learn_rate),
                  metrics=['accuracy'])


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

    # Setup data augmentation
    data_augmentation = Sequential([
        RandomFlip('horizontal'),  # randomly flip images on horizontal edge
        RandomRotation(0.2),  # randomly rotate images by a specific amount
        RandomHeight(0.2),  # randomly adjust the height of an image by a specific  amount
        RandomWidth(0.2),  # randomly adjust the width of an image by a specific amount
        RandomZoom(0.2),  # randomly zoom into an image
        # Rescaling(1./255)  # keep for models like ResNet50V2, remove for EfficientNet
    ], name='data_augmentation')

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
    outputs = Dense(len(train_data_all_10_percent.class_names), activation='softmax', name='output_layer')(x)
    model = Model(inputs, outputs)

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
    loaded_loss, loaded_accuracy = model.evaluate(test_data)
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
    print(pred_classes[:10])

    # Note: This might take a minute or so due to unravelling 790 batches
    y_labels = []
    for images, labels in test_data.unbatch():  # unbatch the test data and get images and labels
        y_labels.append(labels.numpy().argmax())  # append the index which has the largest value (labels are one-hot)
    print(y_labels[:10])  # check what they look like (unshuffled)

    # How many labels are there? (should be the same as how many prediction probabilities we have)
    print(len(y_labels))

    '''Evaluating our models predictions'''

    #


