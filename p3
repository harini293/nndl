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
