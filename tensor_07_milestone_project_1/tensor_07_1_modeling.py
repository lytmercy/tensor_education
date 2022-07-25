import tensorflow as tf
from keras.callbacks import ModelCheckpoint


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

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = 'checkpoints\\milestone_pj\\c.ckpt'  # saving weights requires ".ckpt" extension
    model_checkpoint = ModelCheckpoint(checkpoint_path,
                                       monitor='val_acc',  # save the model weights with best validation accuracy
                                       save_best_only=True,  # only save the best weights
                                       save_weights_only=True,  # only save model weights (not whole model)
                                       verbose=0)  # don't print out whether or not model is being saved

    #

