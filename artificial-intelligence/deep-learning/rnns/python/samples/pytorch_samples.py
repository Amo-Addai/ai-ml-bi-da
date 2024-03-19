import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling2d Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=32)
        
        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)
        
        # Flatten Layer
        self.flatten = nn.Flatten()
        
        # Dense Layer
        self.fc1 = nn.Linear(in_features=32*28*28, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Convolutional Layer
        x = F.relu(self.conv1(x))
        
        # MaxPooling2d Layer
        x = self.pool(x)
        
        # LSTM Layer
        x, _ = self.lstm(x)
        
        # Embedding Layer
        x = self.embedding(x)
        
        # Dropout Layer
        x = self.dropout(x)
        
        # Flatten Layer
        x = self.flatten(x)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x

# Define the model
model = CustomModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming we have data X_train, y_train for training and X_test, y_test for testing
# Fit the model with a good sample of data
def train_model(model, criterion, optimizer, X_train, y_train, epochs=10, batch_size=32):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            inputs = torch.tensor(X_train[i:i+batch_size]).float()
            labels = torch.tensor(y_train[i:i+batch_size]).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, running_loss / len(X_train)))

# Assuming X_test, y_test are the test data
# Evaluate the model
def evaluate_model(model, X_test, y_test, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = torch.tensor(X_test[i:i+batch_size]).float()
            labels = torch.tensor(y_test[i:i+batch_size]).long()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test data: %d %%' % (100 * correct / total))

# Example usage
# train_model(model, criterion, optimizer, X_train, y_train)
# evaluate_model(model, X_test, y_test)


'''
Explanation of Layers:

Convolutional Layer (Conv2d): This layer performs 2D convolution on the input data to extract features from images. It uses a specified number of output channels (filters), kernel size, stride, and padding.

MaxPooling2d Layer: This layer reduces the spatial dimensions of the input data by taking the maximum value within a sliding window (kernel) and moving it over the input feature map.

LSTM Layer: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) layer that is capable of learning long-term dependencies. It maintains a cell state and has gates to control the flow of information.

Embedding Layer: This layer converts integer-encoded inputs into dense vectors of fixed size (embeddings). It is commonly used in natural language processing tasks.

Dropout Layer: Dropout is a regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting.

Flatten Layer: This layer reshapes the input tensor into a 1D tensor, which is required before feeding it into fully connected layers.

Dense Layers (Linear): These layers perform affine transformations on the input data, followed by an activation function. They are the core building blocks of neural networks.

The code also includes training and evaluation functions, along with example usage. The chosen batch size, number of epochs, learning rate, and other parameters can be adjusted based on the specific requirements of the dataset and the model.
'''

