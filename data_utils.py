import dbcollection as dbc
import cv2
import numpy as np
from utils import gaussian_image_map


mpii = dbc.load('mpii_pose')


def ascii2str(array):
    s = ''
    for num in array:
        s += chr(num)
    return s


def get_image_keypoints():
    filenames = mpii.get('train', 'image_filenames')
    keypoints = mpii.get('train', 'keypoints')
    keypoint_ids_per_image = mpii.get('train', 'list_keypoints_per_image')

    for name, keypoint_ids in zip(filenames[:1], keypoint_ids_per_image[:1]):
        heatmap = list()
        img = cv2.imread(ascii2str(name))
        height, width = img.shape[0], img.shape[1]
        indices = np.where(keypoint_ids != -1)[0]
        keypoint_indices = np.take(keypoint_ids, indices)
        for idx in keypoint_indices:
            parts = keypoints[idx]
            for part in parts:
                if part[-1] != -1:
                    # cv2.circle(img, (int(part[0]), int(part[1])), 1, (255, 0, 0), 2)
                    gaussian_map = gaussian_image_map(height, width, int(part[0]), int(part[1]), 3.0)
                    heatmap.append(gaussian_map)

        heatmap = np.array(heatmap)
        heatmap = np.max(heatmap, axis=0)
        heatmap = np.expand_dims(heatmap, axis=-1)
        heatmap = heatmap * 255.0
        heatmap = heatmap.astype(np.uint8)
        img += heatmap
        print(heatmap.shape)
        cv2.imshow('window', img)
        # cv2.imshow('heatmap', heatmap)
        cv2.waitKey(0)


get_image_keypoints()
