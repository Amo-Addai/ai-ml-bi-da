import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

model = Sequential()

# Layer 1: Convolutional layer with 32 filters, kernel size of (3,3), input shape (assuming input is 2D image)
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2: Convolutional layer with 64 filters, kernel size of (3,3)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3: Convolutional layer with 128 filters, kernel size of (3,3)
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4: Flatten layer to convert 2D matrix data to a vector
model.add(Flatten())

# Layer 5: Dense layer with 512 neurons
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Layer 6: Dense layer with 256 neurons
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 7: Dense layer with 128 neurons
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8: Dense layer with 64 neurons
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 9: Dense layer with 32 neurons
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 10: Output layer with 10 neurons (assuming 10 classes)
model.add(Dense(10))
model.add(Activation('softmax'))  # Softmax activation for multi-class classification

# Compile the model
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Summary of the model architecture
model.summary()

'''
Explanation of each layer:

Conv2D: Convolutional layer extracts features from the input image using filters. The number of filters determines the number of output channels.
Activation: Activation function introduces non-linearity to the model. ReLU (Rectified Linear Activation) is used here.
MaxPooling2D: Max pooling reduces the spatial dimensions of the feature maps, reducing computation and controlling overfitting.
Flatten: Flatten layer converts the 2D feature maps into a 1D vector, necessary for feeding into the Dense layers.
Dense: Fully connected layer where each neuron is connected to every neuron in the previous layer.
Dropout: Dropout layer randomly sets a fraction of input units to zero during training to prevent overfitting.
Now, let's fit the model with sample data:
'''

# Assuming X_train and y_train are input images and labels for training
# Assuming X_test and y_test are input images and labels for testing

# Fit the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

'''
Explanation of parameters:

batch_size: Number of samples per gradient update. It defines the number of samples that will be propagated through the network.
epochs: Number of epochs to train the model. An epoch is one pass through the entire training dataset.
validation_split: Fraction of the training data to be used as validation data.
After training, we can evaluate the model on test data:
'''

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, batch_size=64)

'''
Explanation of evaluation parameters:

batch_size: Number of samples per evaluation step.
This model architecture is a deep CNN with multiple convolutional layers followed by fully connected layers. It's suitable for image classification tasks and includes techniques like dropout to prevent overfitting.
'''

