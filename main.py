import os
# remove info and warning messages to remove GPU spam
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Since the class names are not included with the dataset, store them here to use later when plotting the images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# There are 60.000 images, each 28 x 28 px
print(train_images.shape)
# 28 arrays each containing 28 values ranging from 0 to 255
print(train_images[0])
# There are also 60.000 labels ranging from 0-9 ( see class_names )
print(len(train_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# we need to preprocess train and test images to scale values between 0 and 1 so we'll divide them by 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# to check we'll display 25 first preprocessed images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    # flatten will transform 2 dim array to 1 dim array -> 28 x 28 goes to one row of 786 pixels
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# compile the model before the training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model on train data
model.fit(train_images, train_labels, epochs=10)

# compare how trained model performs vs test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# attaching the softmax layer to covert the logits to probabilities,
# Softmax will predict probability for each logit and to what possible class it could belong to
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# try to predict on test images
predictions = probability_model.predict(test_images)

# As there are 10 possible clothes classes prediction[0] will show possibility of logit belonging to each - array of 10 elements
print(predictions[0])
print('The image in the prediction is probably clothing type of: ' +
      class_names[np.argmax(predictions[0])])
