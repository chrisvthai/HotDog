import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import numpy as np
import argparse
import sys
from sklearn import preprocessing
from read_data import prepare_data,read_image_array,read_single_image

def main(_):
	saver = tf.train.Saver()
    path = "./hotdog-model/"
    if not os.path.exists(path):
      os.makedirs(path)

    with tf.Session() as sess:

      sess.run(tf.global_variables_initializer())
      saver.restore(sess, path + 'test-model')
      file_list, y_image_label = prepare_data(FLAGS.image_dir)

      le = preprocessing.LabelEncoder()
      y_one_hot = tf.one_hot(le.fit_transform(y_image_label),depth=2)

      x_feed = sess.run(read_image_array(file_list))
      y_feed = sess.run(y_one_hot)

      for i in range(1):
        if i % 10 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x:x_feed , y_: y_feed})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x:x_feed , y_: y_feed})
        save_path = saver.save(sess, path + 'test-model')

      predicted = tf.argmax(y_conv, 1)

      if FLAGS.predict_image != "":
        x_single_img = sess.run(read_single_image(FLAGS.predict_image))
        print('You got %s'%le.inverse_transform(sess.run(predicted,feed_dict={x:x_single_img}))[0])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--predict_image',
        type=str,
        default="",
        help='Unknown image'
    )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)