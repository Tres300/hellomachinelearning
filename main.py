import os
import cv2
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

# Get handwritten digit dataset from keras (convenient!)
mnist = tf.keras.datasets.mnist

# x_train - image training
# y_train - true digit training
# x_test - image testing
# y_test - true digit testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Model Training
# # Normalize image data to be in range 0-1
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# # Neural network model
# model = tf.keras.models.Sequential()
#
# # Add layers
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
#
# model.save('handwrittenDigitModel')

# Model Loading/Testing
model = tf.keras.models.load_model('handwrittenDigitModel')

loss, accuracy = model.evaluate(x_test, y_test)

print("Loss: {}".format(loss))
print("Accuracy: {}".format(accuracy))
