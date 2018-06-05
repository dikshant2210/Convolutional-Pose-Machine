import tensorflow as tf
import cv2
import numpy as np
from model import ConvolutionalPoseMachine as CPM
from utils import grid_plot


def predict(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (368, 368))
    image = np.expand_dims(image, axis=0)

    model = CPM(stages=3, joints=16)
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        saver.restore(sess, save_path='model/weights/model.ckpt-3001')
        heatmaps = sess.run([model.heatmaps], feed_dict={model.images: image})

    heatmaps = np.array(heatmaps)
    heatmaps = np.squeeze(heatmaps)
    grid_plot(heatmaps[2, :, :, :])


if __name__ == '__main__':
    path = 'images/test_image.JPG'
    predict(path)
