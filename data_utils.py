import dbcollection as dbc
import cv2
import numpy as np
from utils import gaussian_image_map, plot_heatmap


mpii = dbc.load('mpii_pose')
filenames = mpii.get('train', 'image_filenames')
keypoints = mpii.get('train', 'keypoints')
objpos = mpii.get('train', 'objpos')
scale = mpii.get('train', 'scale')
keypoint_ids_per_image = mpii.get('train', 'list_keypoints_per_image')


def ascii2str(array):
    s = ''
    for num in array:
        s += chr(num)
    return s


def create_heatmap(image, person_keypoints):
    height, width = image.shape[0], image.shape[1]
    heatmap = list()
    for part in person_keypoints:
        if part[-1] != -1:
            gaussian_map = gaussian_image_map(height, width, int(part[0]), int(part[1]), 4.0)
            heatmap.append(gaussian_map)
        else:
            heatmap.append(np.zeros((height, width)))
    heatmap = np.array(heatmap)
    heatmap = np.moveaxis(heatmap, 0, -1)
    return heatmap


def train_generator():
    file_index = list()
    for index, val in enumerate(keypoint_ids_per_image):
        num_persons = len(np.where(val != -1)[0])
        for _ in range(num_persons):
            file_index.append(index)

    for index in range(len(file_index)):
        image_path = ascii2str(filenames[file_index[index]])
        person_keypoints = keypoints[index]

        image = cv2.imread(image_path)

        person_keypoints = person_keypoints[np.where((person_keypoints[:, 2] != -1) *
                                                     (person_keypoints[:, 0] != 0) * (person_keypoints[:, 1] != 0))]
        x_start = max(min(person_keypoints[:, 1]) - 100, 0)
        x_end = min(max(person_keypoints[:, 1]) + 100, image.shape[0])
        y_start = max(min(person_keypoints[:, 0]) - 100, 0)
        y_end = min(max(person_keypoints[:, 0]) + 100, image.shape[1])

        x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)

        person_keypoints[:, 0] = person_keypoints[:, 0] - y_start
        person_keypoints[:, 1] = person_keypoints[:, 1] - x_start

        crop_image = image[x_start:x_end, y_start:y_end, :]
        height, width = crop_image.shape[0], crop_image.shape[1]
        person_keypoints[:, 0] = person_keypoints[:, 0] * 368.0 / width
        person_keypoints[:, 1] = person_keypoints[:, 1] * 368.0 / height

        crop_image = cv2.resize(crop_image, (368, 368))
        crop_heatmap = create_heatmap(crop_image, person_keypoints)
        plot_heatmap(crop_image, crop_heatmap)


if __name__ == '__main__':
    print(filenames.shape)
    print(keypoints.shape)
    print(objpos.shape)
    print(scale.shape)
    print(keypoint_ids_per_image.shape)
    train_generator()
