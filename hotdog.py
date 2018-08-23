import tensorflow as tf
from tensorflow.python.platform import gfile
import keras
import os
import numpy as np
import argparse
import sys
from sklearn import preprocessing
from read_data import prepare_data,read_image_array,read_single_image


x = tf.placeholder(tf.float32, shape=[None, 2352])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

x_image = tf.reshape(x, [-1, 28, 28, 3])

model = keras.models.Sequential()
model.add(keras.layers.Reshape((28, 28, 3), input_shape=(2352,)))
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', data_format="channels_last", activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), padding='same', data_format="channels_last",
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(2, activation='softmax'))


learning_rate = 0.005
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
filepath = 'HotDogModel'

path = 'images'

file_list, y_image_label = prepare_data(path)

le = preprocessing.LabelEncoder()
y_one_hot = tf.one_hot(le.fit_transform(y_image_label),depth=2)

x_feed = read_image_array(file_list)
y_feed = y_one_hot

for i in range(11):
  model.fit(x_feed, y_feed,
    epochs=1,
    steps_per_epoch=500,
    validation_data=(x_feed, y_feed),
    validation_steps=100,
    shuffle=True,
    verbose=True)

  model.save('HotDogModel.h5')
