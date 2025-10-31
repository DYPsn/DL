import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0
# Define MLP model
model = keras.Sequential([
layers.Dense(128, activation='relu', input_shape=(784,)),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
# Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", acc)
