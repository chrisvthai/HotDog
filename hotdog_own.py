import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
from read_data import prepare_data,read_image_array,read_single_image

def main(_):

    # Neural net definition
    x = tf.placeholder(tf.float32, shape=[None, 2352])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    initializer = tf.contrib.layers.xavier_initializer()
    x_reshaped = tf.reshape(x, [-1, 28, 28, 3])

    conv_1 = tf.layers.conv2d(
          inputs=x_reshaped,
          filters=32,
          kernel_size=5,
          padding='same',
          activation=tf.nn.relu,
          kernel_initializer=initializer
          )

    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2,2], strides=2)

    conv_2 = tf.layers.conv2d(
          inputs=pool_1,
          filters=64,
          kernel_size=5,
          padding='same',
          activation=tf.nn.relu,
          kernel_initializer=initializer
          )

    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2,2], strides=2)
    pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])

    dense_1 = tf.layers.dense(inputs=pool_2_flat, units=1024, activation=tf.nn.relu, kernel_initializer=initializer)
    dropout = tf.layers.dropout(inputs=dense_1, rate=0.4)
    y_conv  = tf.layers.dense(inputs=dropout, units=2, kernel_initializer=initializer) 

    # Train step data
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    saver = tf.train.Saver()
    path = "./hotdog-model/"
    if not os.path.exists(path):
      os.makedirs(path)

    with tf.Session() as sess:
      # sess.run(tf.global_variables_initializer())
      saver.restore(sess, path + 'test-model')
      file_list, y_image_label = prepare_data(FLAGS.image_dir)
      le = preprocessing.LabelEncoder()
      y_one_hot = tf.one_hot(le.fit_transform(y_image_label),depth=2)

      if FLAGS.train:

        x_feed = sess.run(read_image_array(file_list))
        y_feed = sess.run(y_one_hot)

        for i in range(80):
          if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:x_feed , y_: y_feed})
            print('step %d, training accuracy %g' % (i, train_accuracy))
          train_step.run(feed_dict={x:x_feed , y_: y_feed})
          save_path = saver.save(sess, path + 'test-model')

      elif FLAGS.predict_image != "":
        predicted = tf.argmax(y_conv, 1)
        x_single_img = sess.run(read_single_image(FLAGS.predict_image))
        image = x_single_img
        image = np.array(image, dtype='int64')
        pixels = image.reshape((28, 28, 3))
        print(pixels)
        plt.imshow(pixels)
        plt.show()
        print('You got %s'%le.inverse_transform(sess.run(predicted,feed_dict={x:x_single_img}))[0])



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--image_dir',
        type=str,
        default='images',
        help='Path to folders of labeled images.'
  )
  parser.add_argument(
        '--predict_image',
        type=str,
        default="",
        help='Unknown image'
    )
  parser.add_argument(
        '--train',
        help='Train the network',
        action='store_true'
        )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)