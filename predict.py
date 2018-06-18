import tensorflow as tf
import cv2
import numpy as np
from model import ConvolutionalPoseMachine as CPM
from utils import grid_plot


keypoint_names = {0: 'right ankle',
                  1: 'right knee',
                  2: 'right hip',
                  3: 'left hip',
                  4: 'left knee',
                  5: 'left ankle',
                  6: 'pelvis',
                  7: 'thorax',
                  8: 'upper neck',
                  9: 'head top',
                  10: 'right wrist',
                  11: 'right elbow',
                  12: 'right shoulder',
                  13: 'left shoulder',
                  14: 'left elbow',
                  15: 'left wrist'}


def plot_keypoints(heatmap, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    height, width = image.shape[0], image.shape[1]
    heatmap_width, heatmap_height = heatmap.shape[0], heatmap.shape[1]
    for count, array in enumerate(heatmap.T):
        conf_score = np.max(array)
        if conf_score < 0.1:
            continue
        x, y = np.where(array == conf_score)
        x, y = x[0] * width / heatmap_width, y[0] * height / heatmap_height
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), 1, (0, 0, 255), 5)
        cv2.putText(image, keypoint_names[count], (x+1, y), 1, 1, (0, 0, 255), 1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)


def predict(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (368, 368))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    model = CPM(stages=3, joints=16)
    model.build_model()
    saver = tf.train.Saver(max_to_keep=None)
    with tf.Session() as sess:
        saver.restore(sess, save_path='model/weights/model.ckpt-1')
        heatmaps = sess.run([model.heatmaps], feed_dict={model.images: image})

    heatmaps = np.array(heatmaps)
    heatmaps = np.squeeze(heatmaps)
    # grid_plot(heatmaps[2, :, :, :])
    plot_keypoints(heatmaps[2, :, :, :], file_path)


if __name__ == '__main__':
    # path = 'images/standing_test.jpg'
    # path = 'images/test_image.jpg'
    path = 'images/upper_body.JPG'
    predict(path)
