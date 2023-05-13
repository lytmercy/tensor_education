from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def compile_model(model):
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'])


def compile_improved_adam_model(model):
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['mae'])


def auto_print_mae(model, number, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Model_{number} loss:{loss}\nModel_{number} mae:{mae}")


def run():
    """
    01.6 Exercises

    1. Create your own regression dataset (or make the one we created in "Create data to view and fit" bigger) and 
    build fit a model to it.
    2. Try building a neural network with 4 Dense layers and fitting it to your own regression dataset, 
    how does it perform?
    3. Try and improve the results we got on the insurance dataset, some things you might want to try include:
        - Building a larger model (how does one with 4 dense layers go?).
        - Increasing the number of units in each layer.
        - Lookup the documentation of Adam and find out what the first parameter is, 
        what happens if you increase it by 10x?
        - What happens if you train for longer (say 300 epochs instead of 200)?
    4. Import the Boston pricing dataset from TensorFlow tf.keras.datasets and model it.
    """

    '''Exercise - 1'''

    # Create a dataset

    # Create numpy array
    numpy_x = np.concatenate([np.arange(-150, -70, 3), np.arange(60, 120, 5),
                              np.arange(-50, 10, 3), np.arange(20, 50, 5),
                              np.arange(130, 150, 3), np.arange(160, 200, 5),
                              np.arange(205, 221, 1), np.arange(230, 250, 5)], axis=0)
    # print(numpy_x)
    print("Size:", numpy_x.size)

    numpy_y = np.concatenate([np.arange(-160, -80, 3), np.arange(50, 110, 5),
                              np.arange(-40, 20, 3), np.arange(30, 60, 5),
                              np.arange(120, 140, 3), np.arange(150, 190, 5),
                              np.arange(215, 231, 1), np.arange(240, 260, 5)], axis=0)
    # print(numpy_y)
    print("Size:", numpy_y.size)

    # Create features (using tensors)
    X = tf.constant(numpy_x, shape=(100,))

    # Create labels
    y = tf.constant(numpy_y, shape=(100,))

    # print(f"X:\n{X}\ny:\n{y}")

    # Plotting data
    plt.scatter(X, y)
    plt.show()

    # Split data to Train\Test dataset
    X_train = X[:80]
    y_train = y[:80]

    X_test = X[80:]
    y_test = y[80:]

    '''Exercise - 2'''

    # Build ex_model (ex_model = exercise_model)

    # Create the model
    ex_model = tf.keras.Sequential([
        tf.keras.layers.Dense(150),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_model(ex_model)

    # Fit the model
    ex_history = ex_model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

    # Plot fitting history
    pd.DataFrame(ex_history.history).plot()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

    '''Exercise - 3'''

    # Read in the insurance dataset
    insurance = pd.read_csv(
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

    # Create Column Transformer
    ct = make_column_transformer(
        (MinMaxScaler(), ['age', 'bmi', 'children']),
        (OneHotEncoder(handle_unknown="ignore"), ['sex', 'smoker', 'region'])
    )

    # Create X & y
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    # Split data on train\test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit column transformer
    ct.fit(X_train)

    # Transform training and test datasets
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)

    # Set random seed
    tf.random.set_seed(42)

    # Build model_3
    insurance_model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_model(insurance_model_3)

    # Fit the model
    insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

    # Evaluate 3rd model
    auto_print_mae(insurance_model_3, 3, X_test_normal, y_test)

    # Improving model

    # 1- Build deeper model
    # Set random seed
    tf.random.set_seed(42)

    # === Build model_4-1 ===
    insurance_model_4_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_model(insurance_model_4_1)

    # Fit the model
    insurance_model_4_1.fit(X_train_normal, y_train, epochs=200, verbose=0)

    # Evaluate 3rd model
    auto_print_mae(insurance_model_4_1, 41, X_test_normal, y_test)

    # 2- Increasing the number of units in each layer (hidden units)
    # Set random seed
    tf.random.set_seed(42)

    # === Build model_4-2 ===
    insurance_model_4_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_model(insurance_model_4_2)

    # Fit the model
    insurance_model_4_2.fit(X_train_normal, y_train, epochs=200, verbose=0)

    # Evaluate 3rd model
    auto_print_mae(insurance_model_4_2, 42, X_test_normal, y_test)

    # 3- Improving Adam optimizer
    # Set random seed
    tf.random.set_seed(42)

    # === Build model_4-3 ===
    insurance_model_4_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_improved_adam_model(insurance_model_4_3)

    # Fit the model
    insurance_model_4_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

    # Evaluate 3rd model
    auto_print_mae(insurance_model_4_3, 43, X_test_normal, y_test)

    # 4- Longer training
    # Set random seed
    tf.random.set_seed(42)

    # === Build model_4-3 ===
    insurance_model_4_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    compile_improved_adam_model(insurance_model_4_4)

    # Fit the model
    insurance_model_4_4.fit(X_train_normal, y_train, epochs=300, verbose=0)

    # Evaluate 3rd model
    auto_print_mae(insurance_model_4_4, 44, X_test_normal, y_test)

    # Comparing results
    _, model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)
    _, model_41_mae = insurance_model_4_1.evaluate(X_test_normal, y_test)
    _, model_42_mae = insurance_model_4_2.evaluate(X_test_normal, y_test)
    _, model_43_mae = insurance_model_4_3.evaluate(X_test_normal, y_test)
    _, model_44_mae = insurance_model_4_4.evaluate(X_test_normal, y_test)

    model_results = [["model_3", model_3_mae],
                     ["model_41_layers", model_41_mae],
                     ["model_42_units", model_42_mae],
                     ["model_43_adam", model_43_mae],
                     ["model_44_epochs", model_44_mae]]
    #
    all_results = pd.DataFrame(model_results, columns=['model', 'mae'])
    # print(all_results)

    '''Exercise - 4'''

    # import Boston pricing dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

    # print(f"X:{X_train}\ny:{y_train}")

    # === Build boston_model ===
    boston_model = tf.keras.Sequential([
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(150),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    compile_improved_adam_model(boston_model)

    # Fit the model
    boston_model.fit(X_train, y_train, epochs=500, verbose=0)

    # Evaluate the model
    auto_print_mae(boston_model, "boston", X_test, y_test)

    # Increasing hidden units and epochs
    boston_model_v2 = tf.keras.Sequential([
        tf.keras.layers.Dense(250),
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(150),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
    ])

    compile_improved_adam_model(boston_model_v2)
    boston_model_v2.fit(X_train, y_train, epochs=700, verbose=0)

    # Evaluate the model
    auto_print_mae(boston_model_v2, "boston_v2", X_test, y_test)
