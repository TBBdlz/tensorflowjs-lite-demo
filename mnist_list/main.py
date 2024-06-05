# How to put ML model into small device
# e.g. Raspberry Pi, Arduino, Smart Phones, etc.
# 1. Write ML model in native language
# 2. Just deploy ML on the web and let the device access it
# 3. Deploy (tflite) lite model on the device

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # type: ignore

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
model.evaluate(test_images, test_labels)

# Save the model in SavedModel format
model.export(filepath='mnist_model')

# Convert the model (assuming you're in the same directory as mnist_model)
converter = tf.lite.TFLiteConverter.from_saved_model('mnist_model')
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
