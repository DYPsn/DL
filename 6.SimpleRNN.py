# Simple RNN on IMDB (minimal working example)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Hyperparameters
vocab_size = 10000    # keep top 10k words
maxlen = 200
# pad/truncate reviews to 200 tokens
embedding_dim = 32
rnn_units = 32
batch_size = 64
epochs = 5
# 1. Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
# 2. Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# 3. Build model
model = keras.Sequential([
layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
layers.SimpleRNN(rnn_units),
# returns last hidden state by default
layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
model.summary()
# 4. Train
history = model.fit(x_train, y_train,
epochs=epochs,
batch_size=batch_size,
validation_split=0.15,
verbose=2)
# 5. Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test loss: {loss:.4f}  —  Test accuracy: {acc:.4f}")
# 6. Quick predictions (first 5 examples)
preds = (model.predict(x_test[:5]) > 0.5).astype("int32").flatten()
print("Predictions:", preds)
print("Ground truth:", y_test[:5])
