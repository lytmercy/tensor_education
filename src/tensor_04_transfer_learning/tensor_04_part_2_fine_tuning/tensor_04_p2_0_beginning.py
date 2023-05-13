import tensorflow as tf
from keras.layers import Dense, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras import applications
from keras import Model

# Import helper functions


# Define Global variable
IMG_SIZE = (224, 224)  # define image size (shape)


def run():
    """04.p2.0 Beginning"""

    '''Creating helper functions'''

    # Get helper_functions.py script from course GitHub
    # wget.download("https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")

    '''10 Food Classes: Working with less data'''

    # Get 10% of the data of the 10 classes
    # unzip_data('10_food_classes_10_percent.zip')

    # Walk through 10 percent data directory and list number of tiles
    # walk_through_dir('10_food_classes_10_percent')

    # Create training and test directories
    train_dir = 'datasets\\10_food_classes_10_percent\\train\\'
    test_dir = 'datasets\\10_food_classes_10_percent\\test\\'

    # Create data input
    train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                                image_size=IMG_SIZE,
                                                                                # this is type of labels
                                                                                label_mode='categorical',
                                                                                # batch_size is 32 by default,
                                                                                # this is generally a good number
                                                                                batch_size=32)

    test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                               image_size=IMG_SIZE,
                                                                               label_mode='categorical')

    # Check the training data datatype
    # print(train_data_10_percent)

    # Check out the class names of our dataset
    # print(train_data_10_percent.class_names)

    # See an example batch of data
    # for images, labels in train_data_10_percent.take(1):
    #     print(f"Images:\n{images}\nLabels: {labels}")

    '''Model 0: Building a transfer learning model using the Keras Functional API
    
    We're going to go through the following steps:
    1. Instantiate a pre-trained base model object by choosing a target model such as EfficientNetB0 
    from tf.keras.applications, setting the include_top parameter to False (we do this because we're going to create our
    own top, which are the output layers for the model).
    2. Set the base model's trainable attribute to False to freeze all of the weights in the pre-trained model.
    3. Define an input layer for our model, for example, what shape of data should our model expect?
    4. [Optional] Normalize the input to our model if it requires. Some computer vision models such as ResNetV250 
    require their input to be between 0 & 1.

    Note: As of writing, the EfficientNet models in the tf.keras.applications module do not require images to be 
    normalized (pixel values between 0 and 1) on input, where as many of the other models do.

    5. Pass the input to the base model.
    6. Pool the outputs of the base model into a shape compatible with the output activation layer (turn base model 
    output tensors into same shape as label tensors). This can be done using tf.keras.layers.GlobalAveragePooling2D() or
    tf.keras.layers.GlobalMaxPooling2D() though the former is more common in practice.
    7. Create an output activation layer using tf.keras.layers.Dense() with the appropriate activation function and 
    number of neurons.
    8. Combine the input and outputs layer into a model using tf.keras.Model().
    9. Compile the model using the appropriate loss function and choose of optimizer.
    10. Fit the model for desired number of epochs and with necessary callbacks (in our case, we'll start off with the 
    TensorBoard callback).
    '''

    # 1. Create base model with tf.keras.applications
    base_model = applications.EfficientNetB0(include_top=False)

    # 2. Freeze the base model (so the pre-learned patterns remain)
    base_model.trainable = False

    # 3. Create input into the base model
    inputs = Input(shape=(224, 224, 3), name='input_layer')

    # 4. If using ResNet50V2, add this to speed up convergence, remove for EfficientNet
    # x = Rescaling(1./255)(input)

    # 5. Pass the input to the base_model (note: using tf.keras.applications,
    # EfficientNet input don't have to be normalized)
    x = base_model(inputs)
    # Check data shape after passing it to base_model
    # print(f"Shape after base_model: {x.shape}")

    # 6. Average pool the outputs of the base model (aggregate all the most
    # important information, reduce number of computations)
    x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
    # print(f"After GlobalAveragePooling2D(): {x.shape}")

    # 7. Create the output activation layer
    outputs = Dense(10, activation='softmax', name='output_layer')(x)

    # 8. Combine the input with the outputs into a model
    model_0 = Model(inputs, outputs)

    # 9. Compile the model
    model_0.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])

    # 10. Fit the model (we use fewer steps for validation, so it's faster)
    # history_10_percent = model_0.fit(train_data_10_percent,
    #                                  epochs=5,
    #                                  steps_per_epoch=len(train_data_10_percent),
    #                                  validation_data=test_data_10_percent,
    #                                  # Go through less of the validation data so
    #                                  # epochs are faster (we want faster experiments!)
    #                                  validation_steps=int(0.25 * len(test_data_10_percent)),
    #                                  # Track our model's training logs for visualization later
    #                                  callbacks=[create_tensorboard_callback('transfer_learning',
    #                                                                         '10_percent_feature_extract')])

    # Check layers in our base model
    # for layer_number, layer in enumerate(base_model.layers):
    #     print(layer_number, layer.name)

    # base_model.summary()

    # Check summary of model constructed with Functional API
    # model_0.summary()

    # Check out our model's training curves
    # plot_loss_curves(history_10_percent)
    # plt.show()

    '''Getting a feature vector from a trained model'''

    # The tf.keras.layers.GlobalAveragePooling2D() layer transforms a 4D tensor
    # into a 2D tensor by averaging the values across the inner-axes
    print("\n=== Check GlobalAveragePooling2D layer ===\n")

    # Define input tensor shape (same number of dimensions as the output of efficientnetb0)
    input_shape = (1, 4, 4, 3)

    # Create a random tensor
    tf.random.set_seed(42)
    input_tensor = tf.random.normal(input_shape)
    print(f"Random input tensor:\n {input_tensor}\n")

    # Pass the random tensor through a global average pooling 2D layer
    global_average_pooled_tensor = GlobalAveragePooling2D()(input_tensor)
    print(f"2D global average pooled random tensor:\n {global_average_pooled_tensor}\n")

    # Check the shapes of the different tensors
    print(f"Shape of input tensor: {input_tensor.shape}")
    print(f"Shape of 2D global averaged pooled input tensor: {global_average_pooled_tensor.shape}\n")

    # This is the same as GlobalAveragePooling2D()
    print(f"Same as GlobalAveragePooling2D():\n{tf.reduce_mean(input_tensor, axis=[1, 2])}\n")  # average across
                                                                                              # the middle axes

    '''Practice
    
    GlobalMaxPooling2D - work same as GlobalAveragePooling2D but did so 
    by maxing the input_tensor across the middle two axes.
    '''
    print("=== Check GlobalMaxPooling2D layer ===\n")

    tf.random.set_seed(42)
    input_tensor = tf.random.normal(input_shape)
    print(f"Random input tensor:\n {input_tensor}\n")

    # Pass the random tensor through a global max pooling 2D layer
    global_max_pooled_tensor = GlobalMaxPooling2D()(input_tensor)
    print(f"2D global max pooled random tensor:\n {global_max_pooled_tensor}\n")

    # Check the shapes of the different tensors
    print(f"Shape of input tensor: {input_tensor.shape}")
    print(f"Shape of 2D global max pooled input tensor: {global_max_pooled_tensor.shape}\n")

    # This is the same as GlobalMaxPooling2D()
    print(f"Same as GlobalMaxPooling2D():\n{tf.reduce_max(input_tensor, axis=[1, 2])}")
