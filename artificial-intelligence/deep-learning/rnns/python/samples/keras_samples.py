import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, MaxPooling2D, Activation, Embedding, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import accuracy
from sklearn.model_selection import train_test_split


# Sample data (dummy data for illustration)
X_train = np.random.rand(1000, 50)  # 1000 samples with 50 features
y_train = np.random.randint(2, size=1000)  # Binary classification labels

# Define the model
model_1 = Sequential()

# Embedding layer
model_1.add(Embedding(input_dim=1000, output_dim=64, input_length=50))

# LSTM layers
model_1.add(LSTM(units=128, return_sequences=True))  # First LSTM layer
model_1.add(LSTM(units=64, return_sequences=True))   # Second LSTM layer

# Dropout layer to prevent overfitting
model_1.add(Dropout(0.5))

# Flatten layer
model_1.add(Flatten())

# Dense (fully connected) layers
model_1.add(Dense(units=64, activation='relu'))  # First dense layer
model_1.add(Dense(units=32, activation='relu'))  # Second dense layer

# Output layer with sigmoid activation for binary classification
model_1.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model_1.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Sample test data (dummy data for illustration)
X_test = np.random.rand(200, 50)  # 200 samples with 50 features
y_test = np.random.randint(2, size=200)  # Binary classification labels

# Evaluate the model on test data
loss, accuracy = model_1.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


'''
Explanation of the layers and model components:

Embedding Layer: Converts input sequences into dense vectors of fixed size. It is often used for word embeddings in natural language processing tasks. In this example, it's used to convert integer-encoded words into dense vectors.

LSTM Layers: Long Short-Term Memory (LSTM) layers are a type of recurrent neural network (RNN) layer that can learn long-term dependencies in sequence data. Multiple LSTM layers are stacked to capture complex temporal patterns in the data.

Dropout Layer: Dropout is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting.

Flatten Layer: Flattens the input to a one-dimensional array. This is necessary when transitioning from convolutional or recurrent layers to dense layers.

Dense (Fully Connected) Layers: These are traditional neural network layers where each neuron is connected to every neuron in the preceding layer. They help in learning complex patterns in the data.

Output Layer: This layer produces the final output of the model. In binary classification tasks, a single neuron with sigmoid activation is typically used to output probabilities.

Compilation: In this step, we specify the optimizer, loss function, and evaluation metrics for the model. In this example, we use the Adam optimizer and binary crossentropy loss for binary classification.

Training: The model is trained using the fit() method. We specify parameters like batch size, number of epochs, and validation split. The model learns to minimize the specified loss function on the training data.

Test Data Evaluation: After training, the model is evaluated on test data using the evaluate() method. This provides insights into how well the model generalizes to unseen data.

Adjustments to batch size, number of epochs, and other parameters depend on the specific dataset and problem at hand. Smaller batch sizes may result in slower convergence but better generalization, while larger batch sizes may lead to faster convergence but poorer generalization. The number of epochs should be chosen such that the model doesn't overfit or underfit the training data. Validation split is used to monitor the model's performance on a portion of the training data during training to prevent overfitting.
'''


# Generate dummy data
X = np.random.rand(1000, 50, 10)  # 1000 samples, each with 50 time steps and 10 features
y = np.random.randint(2, size=1000)  # Binary classification labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model_2 = Sequential()

# Add layers
model_2.add(LSTM(units=64, return_sequences=True, input_shape=(50, 10)))  # LSTM layer with 64 units, returns sequences
model_2.add(Conv2D(filters=32, kernel_size=(3, 3)))  # Convolutional layer with 32 filters, 3x3 kernel size
model_2.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer with 2x2 pool size
model_2.add(Activation('relu'))  # ReLU activation function
model_2.add(Embedding(input_dim=1000, output_dim=64))  # Embedding layer, input_dim=1000 (vocab size), output_dim=64
model_2.add(Dropout(0.5))  # Dropout layer with dropout rate of 0.5
model_2.add(Flatten())  # Flatten layer to convert 2D output to 1D
model_2.add(Dense(units=64, activation='relu'))  # Dense layer with 64 units and ReLU activation
model_2.add(Dense(units=32, activation='sigmoid'))  # Dense layer with 32 units and sigmoid activation
model_2.add(Dense(units=1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

# Compile the model
model_2.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy, metrics=[accuracy])

# Fit the model
model_2.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model on test data
loss, acc = model_2.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)


'''
Explanation of the layers and components:

LSTM Layer: Long Short-Term Memory (LSTM) layer is used for sequential data processing, capable of learning long-term dependencies.

Conv2D Layer: Convolutional layer performs convolution operation over the input, useful for learning spatial features in 2D data.

MaxPooling2D Layer: Max pooling layer downsamples the input representation, reducing its dimensionality and computational complexity.

Activation Layer: Applies an activation function to the output of the previous layer. In this case, ReLU activation is used to introduce non-linearity.

Embedding Layer: Maps discrete input values (such as words) to dense vectors of fixed size, often used in natural language processing tasks.

Dropout Layer: Regularization technique that randomly sets a fraction of input units to zero during training, helping prevent overfitting.

Flatten Layer: Converts the input into a one-dimensional array, typically used when transitioning from convolutional or recurrent layers to fully connected layers.

Dense Layers: Fully connected layers where each neuron is connected to every neuron in the preceding layer.

Output Layer: Produces the final output of the model. Sigmoid activation function is used for binary classification tasks.

Compilation: Specifies the optimizer, loss function, and metrics for training the model.

Training: The model is trained on the training data with a specified batch size and number of epochs. Validation split is used to monitor the model's performance on a portion of the training data during training to prevent overfitting.

Evaluation: The trained model is evaluated on test data to assess its performance on unseen data. Test loss and accuracy are computed.

Adjustments to batch size, learning rate, and other parameters depend on the specific dataset and problem at hand. Smaller batch sizes may result in slower convergence but better generalization, while larger batch sizes may lead to faster convergence but poorer generalization. The learning rate determines the step size taken during optimization and affects the speed and stability of training.
'''

