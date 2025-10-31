import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# OR Gate dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1])   # OR output
# Define single-layer perceptron
model = keras.Sequential([
layers.Dense(1, activation='sigmoid', input_shape=(2,))
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# Train model
model.fit(X, y, epochs=500, verbose=0)
# Evaluate
loss, acc = model.evaluate(X, y, verbose=0)
print("Accuracy:", acc)
# Predictions
print("Predictions:", model.predict(X).round())
