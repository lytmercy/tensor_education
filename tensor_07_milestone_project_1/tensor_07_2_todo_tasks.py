import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.layers import Input, Dense, GlobalAveragePooling2D, Activation, Rescaling
from keras.optimizers import Adam
from keras import Model
from keras.models import clone_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import mixed_precision

import numpy as np

# Import helper functions
from helper_functions import create_tensorboard_callback
# Import function for getting data
from tensor_07_milestone_project_1.tensor_07_0_preprocess_data import Dataset

INPUT_SHAPE = (224, 224, 3)


def run():
    """07.2 ToDos Tasks"""

    # === Getting data ===

    # Create class instance
    dataset_instance = Dataset()
    # Load dataset
    dataset_instance.load_dataset()
    # Preprocess datasets
    dataset_instance.preprocess_dataset()

    # Getting train & test data
    train_data = dataset_instance.train_data
    test_data = dataset_instance.test_data

    # Getting class names
    class_names = dataset_instance.ds_info['label'].names

    # Setup mixed precision policy
    # mixed_precision.set_global_policy(policy='mixed_float16')

    # === Build feature extraction model ===

    # Initiate base_model from EfficientNetB0
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    # Create Input layer
    inputs = Input(shape=INPUT_SHAPE, name='input_layer')

    # Build model with Keras functional API
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_average_pool_layer')(x)
    # x = Dense(len(class_names), name='dense_layer')
    # outputs = Activation('softmax', dtype=tf.float32, name='softmax_output_activation')(x)
    outputs = Dense(len(class_names), activation='softmax', name='dense_output_layer')(x)
    own_todo_model = Model(inputs, outputs)

    # Check the model summary
    own_todo_model.summary()

    # Compile model
    own_todo_model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=Adam(),
                           metrics=['accuracy'])

    # === Build callbacks ===

    # Create ModelCheckpoint callbacks
    checkpoint_path = 'checkpoints\\own_todo_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)
    # Create EarlyStopping callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3)  # for monitoring val_loss per 3 epochs
    # Create Reduce Learning rate callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,  # for multiplying learning rate by 0.2 (reduce by x5)
                                  patience=2,
                                  verbose=1,
                                  min_lr=1e-7)

    '''TODO: Fit the feature extraction model'''

    # Fit the feature extraction model for 3 epochs with tensorboard and model checkpoint callbacks
    own_todo_history = own_todo_model.fit(train_data,
                                          epochs=3,
                                          steps_per_epoch=len(train_data),
                                          validation_data=test_data,
                                          validation_steps=int(.15 * len(test_data)),
                                          callbacks=[create_tensorboard_callback('training_logs',
                                                                                 'own_todo_feature_extract_model'),
                                                     model_checkpoint])

    # Evaluate model (unsaved version) on whole test dataset
    results_own_todo_model = own_todo_model.evaluate(test_data)

    '''TODO: Save the whole model to file'''

    # Save model locally
    own_todo_model.save('models\\own_todo_feature_extract_model\\')

    # Load model previously saved above
    loaded_own_todo_model = load_model('models\\own_todo_feature_extract_model\\')

    # Check the layers in the base model and see what dtype policy they're usnig
    for layer in own_todo_model.layers[1].layers[:20]:
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    # Check loaded model performance (this should be same as results_feature extract_model)
    results_loaded_own_todo_model = loaded_own_todo_model.evaluate(test_data)
    print(results_loaded_own_todo_model)

    assert np.isclose(results_own_todo_model, results_loaded_own_todo_model).all()

    '''TODO: Preparing model's layers for fine-tuning'''

    # Load and evaluate downloaded GS model
    loaded_gs_model = load_model('models\\07_efficientnetb0_feature_extract_model_mixed_precision')

    # How does the loaded model perform? (evaluate it on the test dataset)
    results_loaded_gs_model = loaded_gs_model.evaluate(test_data)
    # print(results_loaded_gs_model)

    # Set all of the layers .trainable variable in the loaded model to True (so they're unfrozen)
    loaded_gs_model.trainable = True

    # Check to see what dtype_policy of the layers in your loaded model are
    # for layer in loaded_gs_model.layers:
    #     print(layer.name, layer.trainable, layer.dtype_policy)

    # for layer in loaded_gs_model.layers[1].layers[:20]:
    #     print(layer.name, layer.trainable, layer.dtype_policy)

    # Create ModelCheckpoint callback to save best model during fine-tuning
    checkpoint_path = 'checkpoints\\own_todo_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)

    # Compile the model ready for fine-tuning
    # Use the Adam optimizer with a 10x lower than default learning rate
    loaded_gs_model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=Adam(learning_rate=0.0001),
                            metrics=['accuracy'])

    # Start to fine-tune (all layers)
    # Use 100 epochs as the default
    # Validate on 15% of the test_data
    # Use the create_tensorboard_callback, ModelCheckpoint and EarlyStopping callbacks
    loaded_gs_history = loaded_gs_model.fit(train_data,
                                            epochs=100,
                                            steps_per_epoch=len(train_data),
                                            validation_data=test_data,
                                            validation_steps=int(.15 * len(test_data)),
                                            callbacks=[create_tensorboard_callback('training_log',
                                                                                   'loaded_todo_gs_fine_tuning_model'),
                                                       model_checkpoint,
                                                       early_stopping])

    # Save model locally
    loaded_gs_model.save('models\\loaded_todo_gs_fine_tuning_model\\')

    # Evaluate mixed precision trained fine-tuned model (this should beat DeepFood's 77.4% top-1 accuracy)
    load_todo_gs_model = load_model('models\\loaded_todo_gs_fine_tuning_model')

    results_loaded_todo_gs_model = load_todo_gs_model.evaluate(test_data)
    print(results_loaded_todo_gs_model)

    '''TODO: View training results on TensorBoard'''

    # Run this script in console
    # tensorboard dev upload --logdir ./training_logs --name "Own Fine-tuning Model" --one_shot

    '''TODO: Evaluate your trained model'''

    #




