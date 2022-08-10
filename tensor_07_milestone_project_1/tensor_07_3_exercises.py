import tensorflow as tf
from keras.applications import EfficientNetB4
from keras.layers import Dense, Input
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import helper functions
from helper_functions import create_tensorboard_callback, make_confusion_matrix
# Import function for getting data
from tensor_07_milestone_project_1.tensor_07_0_preprocess_data import Dataset
# Import function for autolabel (modified version of barchart_demo from matplotlib)
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import autolabel
# Import function for visualise and predict from datasets
from tensor_04_transfer_learning.tensor_04_part_2_fine_tuning.tensor_04_p2_2_exercises import dataset_visualise_and_predict
# Import function for visualise and predict from files
from tensor_04_transfer_learning.tensor_04_part_3_scaling_up.tensor_04_p3_1_big_dog_model import file_visualise_and_predict

# Import function for building model & compile model
from tensor_07_milestone_project_1.tensor_07_2_todo_tasks import build_model, compile_model


def run():
    """07.3 Exercises

    1. Use the same evaluation techniques on the large-scale Food Vision model as you did in the previous notebook
    (Transfer Learning Part 3: Scaling up). More specifically, it would be good to see:
        A confusion matrix between all of the model's predictions and true labels.
        A graph showing the f1-scores of each class.
        A visualization of the model making predictions on various images and comparing the predictions to the ground truth.
            For example, plot a sample image from the test dataset and have the title of the plot show the prediction,
            the prediction probability and the ground truth label.
    2. Take 3 of your own photos of food and use the Food Vision model to make predictions on them. How does it go?
    Share your images/predictions with the other students.
    3. Retrain the model (feature extraction and fine-tuning) we trained in this notebook, except this time use
    EfficientNetB4 as the base model instead of EfficientNetB0.
        - Do you notice an improvement in performance?
        - Does it take longer to train?
        - Are there any tradeoffs to consider?
    4. Name one important benefit of mixed precision training, how does this benefit take place?
    """

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

    '''Exercise - 1'''

    own_model = load_model('models\\todo_fine_tuning_model')

    results_loaded_todo_fine_tuning_model = own_model.evaluate(test_data)
    # print(results_loaded_todo_gs_fine_tuning_model)

    # Making prediction
    pred_probs = own_model.predict(test_data, verbose=1)

    # Get the class predictions of each label
    pred_labels = pred_probs.argmax(axis=1)

    y_labels = []
    for images, labels in test_data.unbatch():
        y_labels.append(labels.numpy())

    # 1. Find precision, recall and f1 score

    # Get a dictionary of the classification report
    classification_report_dict = classification_report(y_labels, pred_labels, output_dict=True)

    dict_of_f1_scores = {}

    for k, v in classification_report_dict.items():
        if k == 'accuracy':  # stop once we get to accuracy key
            break
        else:
            dict_of_f1_scores[class_names[int(k)]] = v['f1-score']
    # Turn class dictionary into dataframe for visualization
    f1_scores = pd.DataFrame({'class_name': list(dict_of_f1_scores.keys()),
                           'f1-score': list(dict_of_f1_scores.values())}).sort_values('f1-score', ascending=False)
    print(f1_scores)

    fig, ax = plt.subplots(figsize=(12, 25))
    fig_scores = ax.barh(range(len(f1_scores)), f1_scores['f1-score'].values)
    ax.set_yticks(range(len(f1_scores)))
    ax.set_yticklabels(list(f1_scores['class_name']))
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

    # Visualized and predict on test images
    dataset_visualise_and_predict(own_model, test_data, class_names)
    plt.show()

    '''Exercise - 2'''

    # Prepare own photos
    own_food_images = ['datasets\\own_food_images\\' + img_path
                       for img_path in os.listdir('datasets\\own_food_images')]

    # Make predictions
    file_visualise_and_predict(own_model, own_food_images, class_names)

    '''Exercise - 3'''

    # === Build own model with EfficientNetB4 as base model ===
    efficientnetb4_model = build_model(EfficientNetB4, class_names)

    compile_model(efficientnetb4_model)

    checkpoint_path = 'checkpoints\\own_efficientnet_b4_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=1)

    # Fit model
    history_efficientnetb4_model = efficientnetb4_model.fit(train_data,
                                                            epochs=3,
                                                            steps_per_epoch=len(train_data),
                                                            validation_data=test_data,
                                                            validation_steps=int(.15 * len(test_data)),
                                                            callbacks=[create_tensorboard_callback('training_log',
                                                                                                   'efficientnetb4_101_classes_all_data'),
                                                                       model_checkpoint])

    efficientnetb4_model.save('models\\own_07_efficientnetb4_model\\')

    loaded_efficientnetb4_model = load_model('models\\own_07_efficientnetb4_model')

    # Set trainable for layers to True.
    for layer in loaded_efficientnetb4_model.layers:
        layer.trainable = True

    # Recompile model
    compile_model(loaded_efficientnetb4_model)

    # Create some callbacks for fine-tuning
    checkpoint_path = 'checkpoints\\own_efficientnet_b4_fine_tune_model\\checkpoint.ckpt'
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       save_best_only=True,
                                       verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=2,
                                  verbose=1,
                                  min_lr=1e-7)

    # Fit model for fine-tuning
    history_efficientnetb4_fine_tune_model = loaded_efficientnetb4_model.fit(train_data,
                                                                             epochs=100,
                                                                             steps_per_epoch=len(train_data),
                                                                             validation_data=test_data,
                                                                             validation_steps=int(.15 * len(test_data)),
                                                                             callbacks=[create_tensorboard_callback('training_log',
                                                                                                                    'efficientnetb4_101_classes_all_data_fine_tune'),
                                                                                        model_checkpoint,
                                                                                        early_stopping,
                                                                                        reduce_lr])

    # Evaluate model
    results_efficientnetb4_fine_tune_model = loaded_efficientnetb4_model.evaluate(test_data)
    print(results_efficientnetb4_fine_tune_model)

    '''Exercise - 4
    
    A most important benefit of mixed precision training is -- the performance benefits from float16/bfloat16
    and the numeric stability benefits from float32.
    '''


