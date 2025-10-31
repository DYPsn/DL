#1st code
import tensorflow as tf
device_name = tf.test.gpu_device_name()
print("GPU:", device_name if device_name else "No GPU detected")
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
  print("TPU:", tpu.cluster_spec().as_dict()['worker'])
except:
  print("No TPU detected")


#2nd code
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Define model
model = keras.Sequential([
layers.Flatten(input_shape=(28,28)),
layers.Dense(128, activation='relu'),
layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
# Train
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
