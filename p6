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
