Program 1:
Logistic Regression

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1, W2 = parameters['W1'], parameters['W2']
    A1, A2 = cache['A1'], cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate=0.01):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01):
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(num_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(A2, Y)

        # Backward propagation
        grads = backward_propagation(parameters, cache, X, Y)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 1000 iterations
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return parameters

# Example usage:
input_size = 2
hidden_size = 4
output_size = 1

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Input
Y = np.array([[0, 1, 1, 0]])  # Output

trained_parameters = model(X, Y, input_size, hidden_size, output_size, num_iterations=10000, learning_rate=0.01)



Program 2: Planar data classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize parameters
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def forward_propagation(self, X):
        # Forward pass
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward_propagation(self, X, Y):
        m = X.shape[1]  # Number of samples

        # Backward pass
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * (1 - np.power(self.A1, 2))  # derivative of tanh
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update parameters
        self.W2 -= dW2
        self.b2 -= db2
        self.W1 -= dW1
        self.b1 -= db1

    def train(self, X, Y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # Forward propagation
            predictions = self.forward_propagation(X)

            # Compute cross-entropy loss
            loss = self.cross_entropy_loss(Y, predictions)

            # Backward propagation
            self.backward_propagation(X, Y)

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def cross_entropy_loss(self, Y, A):
        m = Y.shape[1]  # Number of samples
        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

# Load and preprocess the data
X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = X.T
Y = Y.reshape(1, -1)

# Plot the data
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), cmap=plt.cm.Spectral)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Planar Data Classification Dataset')
plt.show()

# Example usage
input_size = 2
hidden_size = 4
output_size = 1

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X, Y, num_epochs=1000, learning_rate=0.01)


program 3:
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)
    def build_model(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1):
        history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history
    def evaluate(self, test_images, test_labels):
        return self.model.evaluate(test_images, test_labels)
    def predict(self, images):
        return self.model.predict(images)
# Create an instance of the NeuralNetwork class
input_shape = (28, 28, 1)
num_classes = 10
nn = NeuralNetwork(input_shape, num_classes)
# Train the neural network
history = nn.train(train_images, train_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = nn.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Make predictions on a few test images
predictions = nn.predict(test_images[:5])

# Plot the first few test images and their predicted labels
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i, :, :, 0], cmap='gray')
    plt.title(f'Predicted: {tf.argmax(predictions[i])}')
    plt.axis('off')
plt.show()


program 4:

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# Define the deep neural network architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
# Make predictions on a few test images
predictions = model.predict(test_images[:5])
# Display the test images and their predicted labels
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {tf.argmax(predictions[i])}, Actual: {tf.argmax(test_labels[i])}')
    plt.show()



program 5:

import tensorflow as tf
from tensorflow.keras import layers, models

# Define helper functions

def create_convolutional_layer(filters, kernel_size, activation='relu', input_shape=None):
   
 Create a convolutional layer with specified parameters.

    if input_shape:
        return layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape)
    else:
        return layers.Conv2D(filters, kernel_size, activation=activation)

def create_maxpooling_layer(pool_size=(2, 2)):

    Create a max pooling layer with specified pool size.
    return layers.MaxPooling2D(pool_size)

def create_dense_layer(units, activation='relu'):

    Create a dense layer with specified number of units and activation function.
    return layers.Dense(units, activation=activation)

# Build ConvNet model

def build_convnet(input_shape, num_classes):

    Build a Convolutional Neural Network model using TensorFlow.
    model = models.Sequential()

    # Convolutional layers
    model.add(create_convolutional_layer(32, (3, 3), input_shape=input_shape))
    model.add(create_maxpooling_layer())
    model.add(create_convolutional_layer(64, (3, 3)))
    model.add(create_maxpooling_layer())

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(create_dense_layer(128))
    model.add(create_dense_layer(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Load dataset (Fashion MNIST as an example)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build and train the model
input_shape = (28, 28, 1)
num_classes = 10
model = build_convnet(input_shape, num_classes)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

program 6:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence	
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
max_len = 500  # Maximum sequence length
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

# Pad sequences to have consistent length
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# Build RNN model

model = Sequential()
model.add(Embedding(max_features, 32))  # Embedding layer
model.add(SimpleRNN(32))  # Simple RNN layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_train, y_train,
          epochs=10,
          batch_size=batch_size,
          validation_split=0.2)
# Evaluate the model
score, acc = model.evaluate(input_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

