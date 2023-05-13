import tensorflow as tf
import numpy as np

# Import class for init data
from src.tensor_01_regression.regression_data_class import RegressionData
# Import function for calculate mae
from src.tensor_01_regression.tensor_01_3_improving_model import mae


def run():
    """01.4 Process model"""

    '''Saving a model'''
    """
    We can save a TensorFlow/Keras model using model.save()
    There are two ways to save a model in TensorFlow:
    - The SavedModel format (default).
    - The HDF5 format.
    """

    # Initialize class instance
    data = RegressionData()
    data.train_test_init()

    # Getting a data
    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test

    # Set random seed
    tf.random.set_seed(42)

    # Creating model
    model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model_2.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['mae'])

    # Fit the model
    model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

    # Save a model using the SavedModel format
    model_2.save('best_model_SavedModel_format')

    # Save a model using the HDF5 format
    model_2.save("best_model_HDF5_format.h5")  # note the addition of '.h5' on the end

    '''Loading a model'''

    # Create prediction and calculate mae for model_2
    model_2_preds = model_2.predict(X_test)
    model_2_mae = mae(y_test, model_2_preds.squeeze()).numpy()

    # Load a model from the SavedModel format
    loaded_saved_model = tf.keras.models.load_model("best_model_SavedModel_format")
    # loaded_saved_model.summary()

    # Compare model_2(model_for_save) with the SavedModel version (should return True)
    # saved_model_preds = loaded_saved_model.predict(X_test)
    # saved_model_mae = mae(y_test, saved_model_preds.squeeze()).numpy()

    # print(f"Saved_MAE:{saved_model_mae}\nModel_2_MAE:{model_2_mae}")
    # print(saved_model_mae == model_2_mae)

    # Load a model from the HDF5 format
    loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")
    loaded_h5_model.summary()

    # Compare model_2 with the loaded HDF5 version (should return True)
    h5_model_preds = loaded_h5_model.predict(X_test)
    h5_model_mae = mae(y_test, h5_model_preds.squeeze()).numpy()

    print(f"HDF5_MAE:{h5_model_mae}\nModel_2_MAE:{model_2_mae}")
    print(h5_model_mae == model_2_mae)
