import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Hyperparameters
vocab_size = 10000
maxlen = 200
embedding_dim = 64
# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
# Pad sequences to fixed length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# Build LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# Train
history = model.fit(x_train, y_train,
                    epochs=3,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=2)
# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {acc:.4f}")
