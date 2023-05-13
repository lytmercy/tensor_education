import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import function for tensorboard callbacks
from src.tensor_04_transfer_learning.tensor_04_part_1_feature_extraction.tensor_04_p1_0_beginning import create_tensorboard_callback
# Import function for creating model
from src.tensor_04_transfer_learning.tensor_04_part_1_feature_extraction.tensor_04_p1_1_tensorhub import create_model

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])


def fit_model(model, experiment_name, train_data, test_data, num_epochs=5):
    return model.fit(train_data,
                     epochs=num_epochs,
                     steps_per_epoch=len(train_data),
                     validation_data=test_data,
                     validation_steps=len(test_data),
                     callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',
                                                           experiment_name=experiment_name)])


def create_binary_model(model_url):
    """Takes a TensrFlow Hub URL and create a Keras Sequential model with it

    :param model_url: (str) a TensorFlow Hub feature extraction URL
    :return: An uncompiled a Keras Sequential model with model_url as feature
    extractor layer and Dense output layer.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extraction_layer',
                                             input_shape=IMAGE_SHAPE+(3,))

    # Create model
    model = Sequential([
        feature_extractor_layer,
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    return model


def run():
    """
    04.p1.2 Exercises

    1. Build and fit a model using the same data we have here but with the MobileNetV2 architecture feature extraction (mobilenet_v2_100_224/feature_vector) from TensorFlow Hub, how does it perform compared to our other models?
    2. Name 3 different image classification models on TensorFlow Hub that we haven't used.
    3. Build a model to classify images of two different things you've taken photos of.
        - You can use any feature extraction layer from TensorFlow Hub you like for this.
        - You should aim to have at least 10 images of each class, for example to build a fridge versus oven classifier, you'll want 10 images of fridges and 10 images of ovens.
    4. What is the current best performing model on ImageNet?
        - Hint: you might want to check sotabench.com for this.
    """

    '''Exercise - 1'''

    # Getting data
    train_dir = 'datasets\\10_food_classes_10_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    datagen = ImageDataGenerator(rescale=1/255.)

    train_data_10_percent = datagen.flow_from_directory(train_dir,
                                                              target_size=IMAGE_SHAPE,
                                                              batch_size=BATCH_SIZE,
                                                              class_mode='categorical')

    test_data = datagen.flow_from_directory(test_dir,
                                                 target_size=IMAGE_SHAPE,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')

    # Define number of classes
    num_class = train_data_10_percent.num_classes

    # === Define model URL ===

    # ResNet 50 V2 feature vector
    resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

    # EfficientNetB0 feature vector (version 1)
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

    # MobileNet V2 feature vector
    mobilenet_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"

    # === Build ResNet 50 V2 model ===
    print("Building ResNet 50 V2 model")

    resnet_model = create_model(resnet_url, num_class)

    compile_model(resnet_model)

    fit_model(resnet_model, 'resnet50V2', train_data_10_percent, test_data)

    # === Build EfficientNetB0 model ===
    print("Building EfficientNetB0 model")

    efficientnet_model = create_model(efficientnet_url, num_class)

    compile_model(efficientnet_model)

    fit_model(efficientnet_model, 'efficientnetB0', train_data_10_percent, test_data)

    # === Build MobileNet V2 model ===
    print("Building MobileNet V2 model")

    mobilenet_model = create_model(mobilenet_url, num_class)

    compile_model(mobilenet_model)

    fit_model(mobilenet_model, 'mobilenetV2', train_data_10_percent, test_data)

    # === Run script in console ===

    # tensorboard dev upload --logdir ./tensorflow_hub/ \
    # --name "EfficientNetB0 vs. ResNet50V2 vs. MobileNetV2" \
    # --description "Comparing three different TF Hub feature extraction models" \
    # " models architectures using 10% of training images" \
    # --one_shot

    '''Exercise - 2
    
    First we haven't used in classification (only feature vector) this model architecture:
    - EfficientNet-B0
    - ResNet V2 50
    - MobileNet V2
    
    The next three model architectures we haven't used it's:
    - Inception V3 (V2, V1)
    - NASNet-A (large)
    - ConvNeXT 
    '''

    '''Exercise - 3'''

    # Getting the data
    own_train_dir = 'datasets\\second_binary_dataset\\train\\'
    own_test_dir = 'datasets\\second_binary_dataset\\test\\'

    own_datagen = ImageDataGenerator(rescale=1/255.,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)

    own_train_data = own_datagen.flow_from_directory(own_train_dir,
                                                     target_size=IMAGE_SHAPE,
                                                     batch_size=1,
                                                     class_mode='binary')

    own_test_data = own_datagen.flow_from_directory(own_test_dir,
                                                    target_size=IMAGE_SHAPE,
                                                    batch_size=1,
                                                    class_mode='binary')

    own_num_class = own_train_data.num_classes - 1

    # === Building Inception resnet V2 model for own dataset ===

    print("Building Inception ResNetV2 model")

    inception_resnet_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5"

    inception_resnet_model = create_binary_model(inception_resnet_url)

    inception_resnet_model.compile(loss='binary_crossentropy',
                                   optimizer=Adam(),
                                   metrics=['accuracy'])

    fit_model(inception_resnet_model, 'inceptionresnetV2', own_train_data, own_test_data)

    # === Building ResNetV2 model for own dataset ===

    print("Building ResNetV2 model")

    own_resnet_model = create_binary_model(resnet_url)

    own_resnet_model.compile(loss='binary_crossentropy',
                             optimizer=Adam(),
                             metrics=['accuracy'])

    fit_model(own_resnet_model, 'resnet50V2', own_train_data, own_test_data)

    # === Run script in console ===
    # tensorboard dev upload --logdir ./tensorflow_hub/ \
    # --name "InceptionResNetV2 vs. ResNet50V2" \
    # --one_shot

    '''Exercise - 4
    
    Now current top 5 best performing model on ImageNet is:
    - CoCa* (finetuned) = 91.0% Accuracy
    - Model soups (BASIC-L) = 90.98% Accuracy
    - Model soups (ViT-G/14) = 90.94% Accuracy
    - CoAtNet-7 = 90.88% Accuracy
    - CoCa* (frozen) = 90.60% Accuracy
    
    *CoCa (Contrastive Captioners)
    '''
