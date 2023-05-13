from keras.utils import image_dataset_from_directory

# Import helper functions
from src.helper_functions import walk_through_dir


class GettingData:

    def __init__(self):
        self.train_dir = 'datasets\\101_food_classes_10_percent\\train\\'
        self.test_dir = 'datasets\\101_food_classes_10_percent\\test\\'

    def get_train_data(self):
        return image_dataset_from_directory(self.train_dir,
                                            label_mode='categorical',
                                            image_size=(224, 224))

    def get_test_data(self):
        return image_dataset_from_directory(self.test_dir,
                                            label_mode='categorical',
                                            image_size=(224, 224),
                                            shuffle=False)  # don't shuffle test data for prediction analysis

    def get_class_names(self):
        return image_dataset_from_directory(self.test_dir,
                                            label_mode='categorical',
                                            image_size=(224, 224),
                                            shuffle=False).class_names


def run():
    """04.p3.0 Beginning"""

    '''Downloading and preprocessing the data'''

    # Download data from Google Storage (already preformatted)
    # wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip')
    # unzip_data('101_food_classes_10_percent.zip')

    # How many images/classes are there?
    walk_through_dir('datasets\\101_food_classes_10_percent')


