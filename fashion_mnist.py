"Solving a computer vision problem. Fashion MNIST is a dataset wtih  different items of clothing. 
"I train a model from a dataset containing 10 different types."
"Using callbacks to avoid hardcoding a number of epochs"

import tensorflow as tf
#import tensorflow.keras as keras

#from tensorflow.examples.tutorials.mnist import input_data

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nLoss is low so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

fashion_mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
#plt.imshow(train_images[50])
#print(train_labels[50])
#print(train_images[50])


image_n = 10
plt.imshow(test_images[image_n])
print(test_images[image_n])
print(test_labels[image_n])

train_images  = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), # Flatten(input_shape = [28,28]) phots are 28x28 pixels, and they are flattened to an array
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs=10, callbacks = [callbacks])

#model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[image_n])
