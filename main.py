import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

cifar10 = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
class_names = []

# need to divide  by 255 for each channel so that we get values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images.dtype)
# print('--------------')
# print(train_images[0][31])
# print('--------------')
# print(train_images[0])
#
# print(len(train_images[0]))
# print(train_images.shape)
# print('********************************')
# print(train_labels[0])
# print(train_labels.shape)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32)),  # images are 32 x 32 pixels size
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # Cifar10 has 10 categories so last layer has 10 nodes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
