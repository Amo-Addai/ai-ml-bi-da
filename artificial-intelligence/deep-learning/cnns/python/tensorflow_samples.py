import numpy as np
import tensorflow as tf

# Define the CNN model
def build_cnn_1(input_shape):
    model = tf.keras.Sequential([
        # Convolutional layer with 32 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer with 64 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer with 128 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten layer
        tf.keras.layers.Flatten(),

        # Dense layer with 512 neurons and ReLU activation
        tf.keras.layers.Dense(512, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Dense layer with 256 neurons and ReLU activation
        tf.keras.layers.Dense(256, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Dense layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Dense layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Dense layer with 32 neurons and ReLU activation
        tf.keras.layers.Dense(32, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Output layer with 10 neurons and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# Define the model
def build_cnn_2(input_shape):
    model = tf.keras.Sequential([
        # Convolutional layer with 32 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolutional layer with 64 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Convolutional layer with 128 filters, kernel size of (3,3), and ReLU activation
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten layer
        tf.keras.layers.Flatten(),

        # Fully connected layer with 512 neurons and ReLU activation
        tf.keras.layers.Dense(512, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Fully connected layer with 256 neurons and ReLU activation
        tf.keras.layers.Dense(256, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Fully connected layer with 128 neurons and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Fully connected layer with 64 neurons and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Fully connected layer with 32 neurons and ReLU activation
        tf.keras.layers.Dense(32, activation='relu'),
        # Dropout layer
        tf.keras.layers.Dropout(0.5),

        # Output layer with 10 neurons and softmax activation (assuming 10 classes)
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


'''
Explanation:

We define the model architecture using the Sequential API in TensorFlow.
We stack convolutional layers, max pooling layers, fully connected layers, and dropout layers.
We specify the input shape of the images.
We compile the model with the Adam optimizer and categorical cross-entropy loss function.
Finally, we print the summary of the model architecture.
This model architecture is similar to the previous ones but implemented using TensorFlow. It's suitable for image classification tasks and includes techniques like dropout to prevent overfitting.
'''


# Build the CNN model
model_1 = build_cnn_1((28, 28, 1))

# Compile the model
model_1.compile()

# Print model summary
model_1.summary()

# Assuming X_train, y_train, X_test, and y_test are provided
# Let's assume X_train and X_test are numpy arrays of shape (num_samples, height, width, channels)
# Let's assume y_train and y_test are numpy arrays of shape (num_samples, num_classes)

# Fit the model
history = model_1.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model_1.evaluate(X_test, y_test, batch_size=64)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


'''
Explanation of parameters:

X_train and y_train: Training data and labels.
X_test and y_test: Test data and labels.
batch_size: Number of samples per gradient update. It defines the number of samples that will be propagated through the network before updating the model's parameters. A smaller batch size consumes less memory but may result in slower training convergence. Common values are 32, 64, or 128.
epochs: Number of epochs to train the model. An epoch is one pass through the entire training dataset. Increasing the number of epochs may improve the model's performance, but it can also lead to overfitting if the model learns noise in the training data.
validation_split: Fraction of the training data to be used as validation data during training. It helps monitor the model's performance on unseen data and prevents overfitting by providing an early stopping mechanism.

'''


# Build the model
model_2 = build_cnn_2((28, 28, 1))

# Compile the model
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model_2.summary()

# Assuming X_train, y_train, X_test, and y_test are provided
# Let's assume X_train and X_test are numpy arrays of shape (num_samples, height, width, channels)
# Let's assume y_train and y_test are numpy arrays of shape (num_samples, num_classes)

# Fit the model
history = model_2.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model_2.evaluate(X_test, y_test, batch_size=64)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)



'''
Explanation:

Batch Size: The number of samples propagated through the network at once during training. A batch size of 64 means that 64 samples will be processed in each training iteration.
Number of Epochs: The number of times the entire dataset is passed through the network during training. Here, we train for 10 epochs.
Validation Split: Fraction of the training data to be used as validation data to monitor the model's performance during training.
Training: We fit the model to the training data using the fit method, specifying batch size and number of epochs.
Test Data Evaluation: After training, we evaluate the model's performance on the test data using the evaluate method, which returns the test

'''

