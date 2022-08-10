import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.layers import Input, Dense, GlobalAveragePooling2D, Activation, Rescaling
from keras.optimizers import Adam
from keras import Model
from keras.models import clone_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import mixed_precision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Import helper functions
from helper_functions import create_tensorboard_callback, make_confusion_matrix
# Import function for getting data
from tensor_07_milestone_project_1.tensor_07_0_preprocess_data import Dataset
# Import function for autolabel (modified version of barchart_demo from matplotlib)
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import autolabel

INPUT_SHAPE = (224, 224, 3)


def build_model(base_model, class_names):

    # Initiate base_model from EfficientNetB0
    based_model = base_model(include_top=False)
    based_model.trainable = False

    # Create Input layer
    inputs = Input(shape=INPUT_SHAPE, name='input_layer')

    # Build model with Keras functional API
    x = based_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_average_pool_layer')(x)
    # x = Dense(len(class_names), name='dense_layer')
    # outputs = Activation('softmax', dtype=tf.float32, name='softmax_output_activation')(x)
    outputs = Dense(len(class_names), activation='softmax', name='dense_output_layer')(x)
    output_model = Model(inputs, outputs)

    return output_model


def compile_model(model):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])


def run():
    """07.2 ToDos Tasks"""

    # === Getting data ===

    # Create class instance
    dataset_instance = Dataset()
    # Load dataset
    dataset_instance.load_dataset()
    # Preprocess datasets
    dataset_instance.preprocess_dataset(batch=13)

    # Getting train & test data
    train_data = dataset_instance.train_data
    test_data = dataset_instance.test_data

    # Getting class names
    class_names = dataset_instance.ds_info.features['label'].names

    # Setup mixed precision policy
    # mixed_precision.set_global_policy(policy='mixed_float16')

    # === Build feature extraction model ===

    own_todo_model = build_model(EfficientNetB0, class_names)

    # Check the model summary
    # own_todo_model.summary()

    # Compile model
    compile_model(own_todo_model)

    # === Build callbacks ===

    # Create ModelCheckpoint callbacks
    checkpoint_path = 'checkpoints\\own_todo_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)
    # Create EarlyStopping callbacks
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5)  # for monitoring val_loss per 3 epochs
    # Create Reduce Learning rate callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,  # for multiplying learning rate by 0.2 (reduce by x5)
                                  patience=1,
                                  verbose=1,
                                  min_lr=1e-7)

    '''TODO: Fit the feature extraction model'''

    # Fit the feature extraction model for 3 epochs with tensorboard and model checkpoint callbacks
    # own_todo_history = own_todo_model.fit(train_data,
    #                                       epochs=3,
    #                                       steps_per_epoch=len(train_data),
    #                                       validation_data=test_data,
    #                                       validation_steps=int(.15 * len(test_data)),
    #                                       callbacks=[create_tensorboard_callback('training_log',
    #                                                                              'own_todo_feature_extract_model'),
    #                                                  model_checkpoint])

    # Evaluate model (unsaved version) on whole test dataset
    # results_own_todo_model = own_todo_model.evaluate(test_data)

    '''TODO: Save the whole model to file'''

    # Save model locally
    # own_todo_model.save('models\\own_todo_feature_extract_model\\')

    # Load model previously saved above
    # loaded_own_todo_model = load_model('models\\own_todo_feature_extract_model')

    # Check the layers in the base model and see what dtype policy they're usnig
    # for layer in own_todo_model.layers[1].layers[:20]:
    #     print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    # Check loaded model performance (this should be same as results_feature extract_model)
    # results_loaded_own_todo_model = loaded_own_todo_model.evaluate(test_data)
    # print(results_loaded_own_todo_model)

    # assert np.isclose(results_own_todo_model, results_loaded_own_todo_model).all()

    '''TODO: Preparing model's layers for fine-tuning'''

    # Load and evaluate downloaded GS model
    loaded_feature_extract_model = load_model('models\\own_todo_feature_extract_model')

    # How does the loaded model perform? (evaluate it on the test dataset)
    # results_loaded_gs_model = loaded_feature_extract_model.evaluate(test_data)
    # print(results_loaded_gs_model)

    # Set all the layers .trainable variable in the loaded model to True (so they're unfrozen)
    loaded_feature_extract_model.trainable = True

    # Check to see what dtype_policy of the layers in your loaded model are
    # for layer in loaded_gs_model.layers:
    #     print(layer.name, layer.trainable, layer.dtype_policy)

    # for layer in loaded_gs_model.layers[1].layers[:20]:
    #     print(layer.name, layer.trainable, layer.dtype_policy)

    # Create ModelCheckpoint callback to save best model during fine-tuning
    checkpoint_path = 'checkpoints\\own_todo_feature_extract_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       verbose=1)

    # Compile the model ready for fine-tuning
    # Use the Adam optimizer with a 10x lower than default learning rate
    loaded_feature_extract_model.compile(loss='sparse_categorical_crossentropy',
                                         optimizer=Adam(learning_rate=0.0001),
                                         metrics=['accuracy'])

    # Start to fine-tune (all layers)
    # Use 100 epochs as the default
    # Validate on 15% of the test_data
    # Use the create_tensorboard_callback, ModelCheckpoint and EarlyStopping callbacks
    # loaded_todo_fine_tune_history = loaded_feature_extract_model.fit(train_data,
    #                                                                  epochs=100,
    #                                                                  steps_per_epoch=len(train_data),
    #                                                                  validation_data=test_data,
    #                                                                  validation_steps=int(.15 * len(test_data)),
    #                                                                  callbacks=[create_tensorboard_callback('training_log',
    #                                                                                                         'loaded_todo_fine_tuning_model'),
    #                                                                             model_checkpoint,
    #                                                                             early_stopping,
    #                                                                             reduce_lr])

    # Save model locally
    loaded_feature_extract_model.save('models\\todo_fine_tuning_model\\')

    # Evaluate mixed precision trained fine-tuned model (this should beat DeepFood's 77.4% top-1 accuracy)
    loaded_todo_fine_tuning_model = load_model('models\\todo_fine_tuning_model')

    results_loaded_todo_fine_tuning_model = loaded_todo_fine_tuning_model.evaluate(test_data)
    # print(results_loaded_todo_gs_fine_tuning_model)

    '''TODO: View training results on TensorBoard'''

    # Run this script in console
    # tensorboard dev upload --logdir ./training_logs --name "Own Fine-tuning Model" --one_shot

    '''TODO: Evaluate your trained model'''
    """
    Some ideas you might want to go through:

    1. Find the precision, recall and f1 scores for each class (all 101).
    2. Build a confusion matrix for each of the classes.
    3. Find your model's most wrong predictions (those with the highest prediction probability but the wrong prediction).

    """
    # Making prediction
    pred_probs = loaded_todo_fine_tuning_model.predict(test_data, verbose=1)

    # Get the class predictions of each label
    pred_labels = pred_probs.argmax(axis=1)

    y_labels = []
    for images, labels in test_data.unbatch():
        y_labels.append(labels.numpy())

    print(accuracy_score(y_labels, pred_labels))

    # 1. Find precision, recall and f1 score
    print(classification_report(y_labels, pred_labels))

    # Get a dictionary of the classification report
    classification_report_dict = classification_report(y_labels, pred_labels, output_dict=True)
    print(classification_report_dict)

    dict_of_scores = {}

    for k, v in classification_report_dict.items():
        if k == 'accuracy':  # stop once we get to accuracy key
            break
        else:
            dict_of_scores[class_names[int(k)]] = (v['precision'],
                                                   v['recall'],
                                                   v['f1-score'])

    print(dict_of_scores)
    # Turn class dictionary into dataframe for visualization
    scores = pd.DataFrame({'class_name': list(dict_of_scores.keys()),
                           'precision': list(dict_of_scores.values())[0],
                           'recall': list(dict_of_scores.values())[1],
                           'f1-score': list(dict_of_scores.values())[2]}).sort_values('f1-score', ascending=False)
    print(scores.head())

    fig, ax = plt.subplots(figsize=(12, 25))
    fig_scores = ax.barh(range(len(scores)), scores['f1-score'].values)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(list(scores['class_name']))
    ax.set_xlabel('f1-scores')
    ax.set_title("F1-Scores for 10 Different Classes")
    ax.invert_yaxis()
    autolabel(fig_scores, ax)
    plt.show()

    # 2. Make confusion matrix
    make_confusion_matrix(y_true=y_labels,
                          y_pred=pred_labels,
                          classes=class_names,
                          figsize=(100, 100),
                          text_size=20,
                          norm=False,
                          savefig=True)

    # 3. Find most wrong prediction
    filepaths = []
    for filepath in test_data.list_files('101_food_classes_10_percent/test/*/*.jpg',
                                         shuffle=False):
        filepaths.append(filepath.numpy())

    most_wrong = pd.DataFrame({'img': filepaths,
                               'y_true': y_labels,
                               'y_pred': pred_labels,
                               'pred_conf': pred_probs.max(axis=1),
                               'y_true_classname': [class_names[i] for i in y_labels],
                               'y_pred_classname': [class_names[i] for i in pred_labels]})

    most_wrong['pred_correct'] = most_wrong['y_true'] == most_wrong['y_pred']

    top_100_wrong = most_wrong[most_wrong['pred_correct'] == False].sort_values('pred_conf', ascending=False)[:100]
    print(top_100_wrong.head(20))



