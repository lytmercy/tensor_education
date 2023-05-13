import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import sklearn libs for data transformation
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def run():
    """01.5 Larger example"""

    # Read in the insurance dataset
    insurance = pd.read_csv(
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

    # Check out the insurance dataset
    # print(insurance.head())

    # Turn all categories into numbers
    insurance_one_hot = pd.get_dummies(insurance)
    # print(insurance_one_hot.head())  # view the converted columns

    # Create X & y values
    X = insurance_one_hot.drop("charges", axis=1)
    y = insurance_one_hot["charges"]
    # print(X.head(), y.head())

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)  # set random state for reproducible splits

    # Set random seed
    # tf.random.set_seed(42)

    # Create a new model (same as model_2 from tensor_01_3_improving_model)
    # insurance_model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(1),
    #     tf.keras.layers.Dense(1)
    # ])

    # Compile the model
    # insurance_model.compile(loss=tf.keras.losses.mae,
    #                         optimizer=tf.keras.optimizers.SGD(),
    #                         metrics=["mae"])

    # Fit the model
    # insurance_model.fit(X_train, y_train, epochs=100, verbose=0)

    # Check the results of the insurance model
    # insurance_model.evaluate(X_test, y_test)

    # Let's try to improve model
    # Set random seed
    tf.random.set_seed(42)

    # Add an extra layer and increase number of units
    insurance_model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),  # 100 units
        tf.keras.layers.Dense(10),  # 10 units
        tf.keras.layers.Dense(1)  # 1 units (important for output layer)
    ])

    # Compile the model
    insurance_model_2.compile(loss=tf.keras.losses.mae,
                              optimizer=tf.keras.optimizers.Adam(),  # Adam works but SGD doesn't
                              metrics=['mae'])

    # Fit the model and save the history (we can plot this)
    history = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

    # Get a summary of the model
    # insurance_model_2.summary()

    # Evaluate our larger model
    # insurance_model_2.evaluate(X_test, y_test)

    # Plot history (also known as a loss curve)
    # pd.DataFrame(history.history).plot()
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.show()

    # Try training for a little longer (100 more epochs)
    history_2 = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

    # Evaluate the model trained for 200 total epochs
    insurance_model_2_loss, insurance_model_2_mae = insurance_model_2.evaluate(X_test, y_test)
    # print(insurance_model_2_loss, insurance_model_2_mae)

    # Plot the model trained for 200 total epochs loss curves
    # pd.DataFrame(history_2.history).plot()
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.show()

    '''Preprocessing data (normalization and standardization)'''
    """
    A common practice when working with neural networks is 
    to make sure all of the data you pass to them is in the range 0 to 1.
    - This practice is called -normalization- (scaling all values from their original range 
    to, e.g. between 0 and 100,000 to be between 0 and 1).
    - There is another process call -standardization- which converts all of your data to unit variance and 0 mean.
    """

    # Check out the data
    # print(insurance.head())

    # Create column transformer (this will help us normalize/preprocess our data)
    ct = make_column_transformer(
        (MinMaxScaler(), ['age', 'bmi', 'children']),  # get all values between 0 and 1
        (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
    )

    # Create X & y
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    # Build our train and test sets (use random state to ensure same split as before)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit column transformer on the training data only (doing so on test data would result in data leakage)
    ct.fit(X_train)

    # Transform training and test data with normalization (MinMaxScaler) and one hot encoding (OneHotEncoder)
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)

    # Non-normalized and non-one-hot encoded data example
    # print(f"X_train:\n{X_train.loc[0]}")

    # Normalized and one-hot encoded example
    # print(f"Normal X_train:\n{X_train_normal[0]}")

    # Notice the normalized/one-hot encoded shape is larger because of the extra columns
    # print(f"Non-normal X_train(shape):{X_train.shape}\nNormalized X_train(shape):{X_train_normal.shape}")
    # print(f"Non-normal y_train(shape):{y_train.shape}\nNon-normal y_test(shape):{y_test.shape}")

    # Set random seed
    tf.random.set_seed(42)

    # Build the model (3 layers, 100, 10, 1 units) same as insurance_model_2
    insurance_model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    insurance_model_3.compile(loss=tf.keras.losses.mae,
                              optimizer=tf.keras.optimizers.Adam(),
                              metrics=['mae'])

    # Fit the model for 200 epochs (same as insurance_model_2)
    insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

    # Evaluate 3rd model
    insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)
    # print(f"Model_3 loss:{insurance_model_3_loss}\nModel_3 mae:{insurance_model_3_mae}")

    # Compare modelling results from non-normalized data and normalized data
    print(f"Model_2 mae:{insurance_model_2_mae}\nModel_3 mae:{insurance_model_3_mae}")

