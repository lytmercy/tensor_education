import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.layers import Input, Dense, GlobalAveragePooling2D, Activation, Rescaling
from keras.optimizers import Adam
from keras import Model
from keras.models import clone_model, load_model
from keras.callbacks import ModelCheckpoint
from keras import mixed_precision

import numpy as np

import wget

# Import helper functions
from helper_functions import create_tensorboard_callback
# Import function for getting data
from tensor_07_milestone_project_1.tensor_07_0_preprocess_data import Dataset


def run():
    """07.1 Modeling"""

    # Create class instance
    dataset_instance = Dataset()
    # Load dataset
    dataset_instance.load_dataset()
    # Preprocess dataset
    dataset_instance.preprocess_dataset()

    # Getting train & test data
    train_data = dataset_instance.train_data
    test_data = dataset_instance.test_data

    # Getting class names
    class_names = dataset_instance.ds_info.features['label'].names

    '''Create modelling callbacks'''

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = 'checkpoints\\milestone_pj\\checkpoint.ckpt'  # saving weights requires ".ckpt" extension
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       monitor='val_accuracy',  # save the model weights with best validation accuracy
                                       save_best_only=True,  # only save the best weights
                                       save_weights_only=True,  # only save model weights (not whole model)
                                       verbose=1)  # don't print out whether or not model is being saved

    '''Setup mixed precision training'''

    # Turn on mixed precision training
    # mixed_precision.set_global_policy(policy="mixed_float16")  # set global policy to mixed precision

    # print(mixed_precision.global_policy())  # should output "mixed_float16"

    '''Build feature extraction model'''

    # Create base model
    input_shape = (224, 224, 3)
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False  # freeze base model layers

    # Create Functional model
    inputs = Input(shape=input_shape, name='input_shape')
    # Note: EfficientNetBX models have rescaling built-in but if your model didn't you could have a layer like below
    # x = Rescaling(1./255)(x)
    x = base_model(inputs, training=False)  # set base_model to inference mode only
    x = GlobalAveragePooling2D(name='pooling_layer')(x)
    outputs = Dense(len(class_names), activation='softmax', name='dense_output')(x)  # want one output neuron per class
    # Separate activation of output layer so we can output float32 activations
    # outputs = Activation('softmax', dtype=tf.float32, name='softmax_float32')(x)
    model_07 = Model(inputs, outputs)

    # Compile the model
    model_07.compile(loss='sparse_categorical_crossentropy',
                     optimizer=Adam(),
                     metrics=['accuracy'])

    # Check out our model
    # model_07.summary()

    '''Checking layer dtype policies (are we using mixed precision?)'''

    # Check the dtype_policy attributes of layers in our model
    # for layer in model_07.layers:
    #     print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)  # Check the dtype policy of layers

    # print('\n')
    # Check the layers in the base model and see what dtype policy they're using
    # for layer in model_07.layers[1].layers[:20]:  # only check the first 20 layers to save output space
    #     print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    '''Fit the feature extraction model'''

    # Fit the model with callback
    history_101_food_classes_feature_extract = model_07.fit(train_data,
                                                            epochs=3,
                                                            steps_per_epoch=len(train_data),
                                                            validation_data=test_data,
                                                            validation_steps=int(0.15 * len(test_data)),
                                                            callbacks=[create_tensorboard_callback('training_logs',
                                                                                                   'efficientnetb0_101_classes_all_data_feature_extract'),
                                                                       model_checkpoint])

    # Evaluate model (unsaved version) on whole test dataset
    results_feature_extract_model = model_07.evaluate(test_data)
    print(results_feature_extract_model)

    '''Load and evaluate checkpoint weights'''

    # Clone the model we created (this resets all weights)
    cloned_model = clone_model(model_07)
    cloned_model.summary()

    # Where are our checkpoints stored?
    print(checkpoint_path)

    # Load checkpointed weights into cloned_model
    cloned_model.load_weights(checkpoint_path)

    # Compile cloned_model (with same parameters as original model)
    cloned_model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=Adam(),
                         metrics=['accuracy'])

    # Evaluate cloned model with loaded weights (should be same score as trained model)
    results_cloned_model_with_loaded_weights = cloned_model.evaluate(test_data)

    # Loaded checkpoint weights should return very similar results to checkpoint weights prior to saving
    # assert np.isclose(results_feature_extract_model, results_cloned_model_with_loaded_weights).all()  # check if all elements in array are close

    # Check the layers in th base model and see what dtype policy they're using
    for layer in cloned_model.layers[1].layers[:20]:  # check only the first 20 layers to save space
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    '''Save the whole model to file'''

    # Save model locally (if you're using Google Colab, your saved model will Colab instance terminates)
    save_dir = 'models\\07_efficientnetb0_feature_extract_model_mixed_precision\\'
    model_07.save(save_dir)

    # Load model previously save above
    loaded_saved_model = load_model(save_dir)

    # Check the layers in the base model and see what dtype policy they're using
    for layer in loaded_saved_model.layers[1].layers[:20]:  # check only the first 20 layers to save output space
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    # Check loaded model performance (this should be the same as results_feature_extract_model)
    results_loaded_saved_model = loaded_saved_model.evaluate(test_data)
    print(results_loaded_saved_model)

    # The loaded model's results should equal (or at least be very close) to the model's results prior to saving
    # Note: this will only work if you've instantiated results variables
    assert np.isclose(results_feature_extract_model, results_loaded_saved_model).all()

    '''Preparing our model's layers for fine-tuning'''

    # Download the saved model from Google Storage
    wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/07_efficientnetb0_feature_extract_model_mixed_precision.zip')
    # Unzip the SavedModel downloaded from Google Storage

    # Load and evaluate downloaded GS model
    loaded_gs_model = load_model('models\\downloaded_gs_model\\07_efficientnetb0_feature_extract_model_mixed_precision')

    # Get a summary of our downloaded model
    loaded_gs_model.summary()

    # How does the loaded model perform?
    results_loaded_gs_model = loaded_gs_model.evaluate(test_data)

    # Are any of the layers in our model frozen?
    for layer in loaded_gs_model.layers:
        layer.trainable = True  # set all layers to trainable
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)  # make sure loaded model is using mixed
                                                                             # precisiion dtype_policy ("mixed_float16")
    # Check the layers in the base model and see what dtype policy they're using
    for layer in loaded_gs_model.layers[1].layers[:20]:
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    '''A couple more callbacks'''

    #


