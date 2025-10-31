import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sample dataset
text = "deep learning is revolutionizing artificial intelligence"
chars = sorted(list(set(text)))
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}
# Prepare dataset
seq_length = 5
x_data, y_data = [], []
for i in range(len(text) - seq_length):
    seq_in = text[i:i+seq_length]
    seq_out = text[i+seq_length]
    x_data.append([char_to_idx[c] for c in seq_in])
    y_data.append(char_to_idx[seq_out])
x_data = np.array(x_data)
y_data = to_categorical(y_data, num_classes=len(chars))
# Build LSTM model
model = Sequential()
model.add(Embedding(len(chars), 50, input_length=seq_length))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(len(chars), activation='softmax'))
# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train
model.fit(x_data, y_data, epochs=100, verbose=1)
# Text generation function
def generate_text(seed_text, length=50):
    result = seed_text
    for _ in range(length):
        x_input = np.array([[char_to_idx[c] for c in result[-seq_length:]]])
        prediction = model.predict(x_input, verbose=0)
        index = np.argmax(prediction)
        result += idx_to_char[index]
    return result
# Generate new text
print(generate_text("deep "))
