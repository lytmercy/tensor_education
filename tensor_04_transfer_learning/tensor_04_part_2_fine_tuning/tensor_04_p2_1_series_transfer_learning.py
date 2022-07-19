import tensorflow as tf
from keras.models import Sequential
from keras import applications
from keras import Model
from keras.layers import Dense, Flatten, Input
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import random
import wget

# Import helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves
from helper_functions import unzip_data, walk_through_dir

# Define global variable
IMG_SIZE = (224, 224)


def compare_histories(original_history, new_history, initial_epochs=5):
    """Compares two model history objects."""

    # Get original history measurements
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']

    print(len(acc))

    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']

    # Combine original history with new history
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']

    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')


def run():
    """04.p2.1 Running a series of transfer learning experiments

    model_1: Use feature extraction transfer learning on 1% of the training data with data augmentation.
    model_2: Use feature extraction transfer learning on 10% of the training data with data augmentation.
    model_3: Use fine-tuning transfer learning on 10% of the training data with data augmentation.
    model_4: Use fine-tuning transfer learning on 100% of the training data with data augmentation.

    All experiments will be done using the EfficientNetB0 model within the tf.keras.applications module.
    To make sure we're keeping track of our experiments, we'll use our create_tensorboard_callback() function
    to log all of the model training logs.
    """

    # Download and unzip data
    # wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip")
    # unzip_data('10_food_classes_1_percent.zip')

    # Create training and test dirs
    train_dir_1_percent = 'datasets\\10_food_classes_1_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_1_percent\\test\\'

    # Walk through 1 percent data directory and list number of files
    # walk_through_dir('datasets\\10_food_classes_1_percent')

    train_data_1_percent = image_dataset_from_directory(train_dir_1_percent,
                                                        label_mode='categorical',
                                                        batch_size=32,  # default
                                                        image_size=IMG_SIZE)

    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='categorical',
                                             batch_size=32,
                                             image_size=IMG_SIZE)

    '''Adding data augmentation right into the model'''

    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomHeight(0.2),
        RandomWidth(0.2),
        # Rescaling(1./255)  # keep for ResNet50V2, remove for EfficientNetB0
    ], name='data_augmentation')

    # View a random image
    # target_class = random.choice(train_data_1_percent.class_names)  # choose a random class
    # target_dir = 'datasets\\10_food_classes_1_percent\\train\\' + target_class  # create the target directory
    # random_image = random.choice(os.listdir(target_dir))  # choose a random image from target directory
    # random_image_path = target_dir + "/" + random_image  # create the chosen random image path
    # img = mpimg.imread(random_image_path)  # read in the chosen target image
    # plt.imshow(img)  # plot the target image
    # plt.title(f"Original random image from class: {target_class}")
    # plt.axis(False)
    # plt.show()

    # Augment the image
    # augmented_img = data_augmentation(tf.expand_dims(img, axis=0))  # data augmentation model requires
    #                                                                 # shape (None, height, width, 3)
    # plt.figure()
    # plt.imshow(tf.squeeze(augmented_img)/255.)  # requires normalization after augmentation
    # plt.title(f"Augmented random image from class: {target_class}")
    # plt.axis(False)
    # plt.show()

    '''Model 1: Feature extraction transfer learning on 1% of the data with data augmentation'''

    # Setup input shape and base model, freezing the base model layers
    input_shape = (224, 224, 3)
    base_model = applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create input layer
    inputs = Input(shape=input_shape, name='input_layer')

    # Add in data augmentation Sequential model as a layer
    x = data_augmentation(inputs)

    # Give base_model inputs (after augmentation) and don't train it
    x = base_model(x, training=False)

    # Pool output features of base model
    x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)

    # Put a dense layer on as the output
    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    # Make a model with inputs and outputs
    model_1 = Model(inputs, outputs)

    # Compile the model
    model_1.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    # Fit the model
    print("Fitting Model 1")
    history_1_percent = model_1.fit(train_data_1_percent,
                                    epochs=5,
                                    steps_per_epoch=len(train_data_1_percent),
                                    validation_data=test_data,
                                    validation_steps=int(0.25 * len(test_data)),  # validation for less steps
                                    # Track model training logs
                                    callbacks=[create_tensorboard_callback('transfer_learning', '1_percent_data_aug')])

    # Check out model summary
    # model_1.summary()

    # Evaluate on the test data
    # results_1_percent_data_aug = model_1.evaluate(test_data)
    # print(results_1_percent_data_aug)

    # How does the model go with a data augmentation layer with 1% of data
    # plot_loss_curves(history_1_percent)
    # plt.show()

    '''Model 2: Feature extraction transfer learning with 10% of data and data augmentation'''

    # Get 10% of the data of the 10 classes
    train_dir_10_percent = 'datasets\\10_food_classes_10_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    # Setup data inputs
    train_data_10_percent = image_dataset_from_directory(train_dir_10_percent,
                                                         label_mode='categorical',
                                                         image_size=IMG_SIZE)

    # Note: the test data is the same as the previous experiment, we could
    # skip creating this, but we'll leave this here to prectice
    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='categorical',
                                             image_size=IMG_SIZE)

    # Setup input shape to our model
    input_shape = (224, 224, 3)

    # Create a frozen base model
    base_model = applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create input and output layers
    inputs = Input(shape=input_shape, name='input_layer')  # create input layer
    x = data_augmentation(inputs)  # augment our training images
    # pass augmented images to base model but keep
    # it in inference mode, so batchnorm layers don't get
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
    outputs = Dense(10, activation='softmax', name='output_layer')(x)
    model_2 = Model(inputs, outputs)

    # Compile
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.001),   # use Adam optimizer with base learning rate
                    metrics=['accuracy'])

    # # === Creating a ModelCheckpoint callback ===

    # Setup checkpoint path
    checkpoint_path = 'checkpoints\\ten_percent_model_checkpoints_weights\\checkpoint.ckpt'

    # Create a ModeCheckpoint callback that saves the model's weights only
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          # set False to save the entire model
                                          save_weights_only=True,
                                          # set to True to save only the best model instead of a model every epoch
                                          save_best_only=False,
                                          save_freq='epoch',  # save every epoch
                                          verbose=1)

    # Fit the model saving checkpoints every epoch
    print("Fitting Model 2")
    initial_epochs = 5
    history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                              epochs=initial_epochs,
                                              validation_data=test_data,
                                              # do less steps per validation (quicker)
                                              validation_steps=int(0.25 * len(test_data)),
                                              callbacks=[create_tensorboard_callback('transfer_learning',
                                                                                     '10_percent_data_aug'),
                                                         checkpoint_callback])

    # Evaluate on the test data
    # results_10_percent_data_aug = model_2.evaluate(test_data)
    # print(results_10_percent_data_aug)

    # Plot model loss curves
    # plot_loss_curves(history_10_percent_data_aug)
    # plt.show()

    # Load in saved model weights and evaluate model
    model_2.load_weights(checkpoint_path)
    # loaded_weights_model_results = model_2.evaluate(test_data)

    # If the results from our native model and the loaded wights are the same, this should output True
    # print(results_10_percent_data_aug == loaded_weights_model_results)

    # Check the difference between the two results
    # print(np.isclose(np.array(results_10_percent_data_aug), np.array(loaded_weights_model_results)))

    # Check the difference between the two results
    # print(np.array(results_10_percent_data_aug) - np.array(loaded_weights_model_results))

    '''Model 3: Fine-tuning an existing model on 10% of the data'''

    # Layers in loaded model
    # print(model_2.layers)

    # for layer in model_2.layers:
    #     print(layer.trainable)

    # model_2.summary()

    # How many layers are trainable in our base model?
    # print(len(model_2.layers[2].trainable_variables))  # layer at index 2 is the EfficientNetB0 layer (the base model)

    # print(len(base_model.trainable_variables))

    # Check which layers are tuneable (trainable)
    # for layer_number, layer in enumerate(base_model.layers):
    #     print(layer_number, layer.name, layer.trainable)

    base_model.trainable = True

    # Freeze all layers except for the
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Recompile the model (always recompile after any adjustments to a model)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),  # lr is 10x lower than before for fine-tuning
                    metrics=['accuracy'])

    # Check which layers are tuneable (trainable)
    # for layer_number, layer in enumerate(base_model.layers):
    #     print(layer_number, layer.name, layer.trainable)

    # print(len(model_2.trainable_variables))

    # Fine tune for another 5 epochs
    fine_tune_epochs = initial_epochs + 5

    # Refit the model (same as model_2 except with more trainable layers)
    print("Fitting Model 3")
    history_fine_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                                   epochs=fine_tune_epochs,
                                                   validation_data=test_data,
                                                   initial_epoch=history_10_percent_data_aug.epoch[-1],
                                                   validation_steps=int(0.25 * len(test_data)),
                                                   callbacks=[create_tensorboard_callback('transfer_learning',
                                                                                          '10_percent_fine_tune_last_10')])

    # Evaluate the model on the test data
    # results_fine_tune_10_percent = model_2.evaluate(test_data)

    # === Create function for compare histories above ^ ===

    # Use function
    # compare_histories(original_history=history_10_percent_data_aug,
    #                   new_history=history_fine_10_percent_data_aug,
    #                   initial_epochs=5)
    # plt.show()

    '''Model 4: Fine-tuning an existing model all of the data'''

    # Setup data directories
    train_dir = 'datasets\\10_food_classes_all_data\\train\\'
    test_dir = 'datasets\\10_food_classes_all_data\\test\\'

    # How many images are we working with now?
    walk_through_dir('datasets\\10_food_classes_all_data')

    # Setup data inputs
    train_data_10_classes_full = image_dataset_from_directory(train_dir,
                                                              label_mode='categorical',
                                                              image_size=IMG_SIZE)

    # Note: this is the same test dataset we've been using for the previous modelling experiments
    test_data = image_dataset_from_directory(test_dir,
                                             label_mode='categorical',
                                             image_size=IMG_SIZE)

    # Evaluate model (this is the fine-tuned 10 percent of data version)
    # print(model_2.evaluate(test_data))

    # print same values from `results_fine_tune_10_percent`
    # print(results_fine_tune_10_percent)

    # Load model from checkpoint, that way we can fine-tune from
    # the same stage the 10 percent data model was fine-tuned from
    model_2.load_weights(checkpoint_path)  # revert model back to saved weights

    # After loading the weights, this should have gone down (no fine-tuning)
    # print(model_2.evaluate(test_data))

    # Check to see if the above two results are the same (they should be)
    # print(results_10_percent_data_aug)

    # Check which layers are tuneable in the whole model
    # for layer_number, layer in enumerate(model_2.layers):
    #     print(layer_number, layer.name, layer.trainable)

    # Check which layers are tuneable in the base model
    # for layer_number, layer in enumerate(base_model.layers):
    #     print(layer_number, layer.name, layer.trainable)

    # Compile
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=Adam(0.0001),  # divide learning rate by 10 for fine-tuning
                    metrics=['accuracy'])

    # Continue to train and fine-tune the model to our data
    print("Fitting Model 4")
    fine_tune_epochs = initial_epochs + 5

    history_fine_10_classes_full = model_2.fit(train_data_10_classes_full,
                                               epochs=fine_tune_epochs,
                                               initial_epoch=history_10_percent_data_aug.epoch[-1],
                                               validation_data=test_data,
                                               validation_steps=int(0.25 * len(test_data)),
                                               callbacks=[create_tensorboard_callback('transfer_learning',
                                                                                      'full_10_classes_fine_tune_last_10')])

    # Evaluate on all the test data
    results_fine_tune_full_data = model_2.evaluate(test_data)
    print(results_fine_tune_full_data)

    # How did fine-tuning go with more data?
    compare_histories(original_history=history_10_percent_data_aug,
                      new_history=history_fine_10_classes_full,
                      initial_epochs=initial_epochs)
    plt.show()

    '''Viewing our experiment data on TensorBoard'''

    # View tensorboard logs of transfer learning modelling experiments (should be 4 models)
    # Upload TensorBoard dev records
    # tensorboard dev upload --logdir ./transfer_learning \
    # --name "Transfer learning experiments" \
    # --description "A series of different transfer learning experiments with varying amounts of data and fine-tuning" \
    # --one_shot  # exits the uploader when upload has finished

    # View previous experiments
    # tensorboard dev list
