import dbcollection as dbc
import cv2
import numpy as np
from utils import gaussian_image_map, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, grid_plot

GRID_PLOT = False

mpii = dbc.load('mpii_pose')
filenames = mpii.get('train', 'image_filenames')
keypoints = mpii.get('train', 'keypoints')
keypoint_ids_per_image = mpii.get('train', 'list_keypoints_per_image')

file_index = list()
for ind, val in enumerate(keypoint_ids_per_image):
    num_persons = len(np.where(val != -1)[0])
    for _ in range(num_persons):
        file_index.append(ind)

data_split_index = int(len(file_index) * 0.2)
file_index = np.array(file_index)
shuffle_indices = np.arange(len(file_index))
np.random.shuffle(shuffle_indices)
keypoints = keypoints[shuffle_indices]
file_index = file_index[shuffle_indices]


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
    while True:
        for start in range(0, len(file_index[:data_split_index]), BATCH_SIZE):
            end = min(start + BATCH_SIZE, data_split_index)
            x_batch, y_batch = list(), list()

            for index in range(start, end):
                image_path = ascii2str(filenames[file_index[index]])
                person_keypoints = keypoints[index]
                temp_keypoints = person_keypoints

                image = cv2.imread(image_path)

                temp_keypoints = temp_keypoints[np.where((person_keypoints[:, 2] != -1) *
                                                         (person_keypoints[:, 0] != 0) *
                                                         (person_keypoints[:, 1] != 0))]
                x_start = max(min(temp_keypoints[:, 1]) - 100, 0)
                x_end = min(max(temp_keypoints[:, 1]) + 100, image.shape[0])
                y_start = max(min(temp_keypoints[:, 0]) - 100, 0)
                y_end = min(max(temp_keypoints[:, 0]) + 100, image.shape[1])

                x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)

                person_keypoints[:, 0] = person_keypoints[:, 0] - y_start
                person_keypoints[:, 1] = person_keypoints[:, 1] - x_start

                crop_image = image[x_start:x_end, y_start:y_end, :]
                height, width = crop_image.shape[0], crop_image.shape[1]
                person_keypoints[:, 0] = person_keypoints[:, 0] * 46.0 / width
                person_keypoints[:, 1] = person_keypoints[:, 1] * 46.0 / height

                crop_image = cv2.resize(crop_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                heatmap_image = cv2.resize(crop_image, (46, 46))
                crop_heatmap = create_heatmap(heatmap_image, person_keypoints)
                x_batch.append(crop_image)
                y_batch.append(crop_heatmap)
                if GRID_PLOT:
                    grid_plot(crop_heatmap)
            x_batch = np.array(x_batch) / 255.0
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(data_split_index, len(file_index), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(file_index))
            x_batch, y_batch = list(), list()

            for index in range(start, end):
                image_path = ascii2str(filenames[file_index[index]])
                person_keypoints = keypoints[index]
                temp_keypoints = person_keypoints

                image = cv2.imread(image_path)

                temp_keypoints = temp_keypoints[np.where((person_keypoints[:, 2] != -1) *
                                                         (person_keypoints[:, 0] != 0) *
                                                         (person_keypoints[:, 1] != 0))]
                x_start = max(min(temp_keypoints[:, 1]) - 100, 0)
                x_end = min(max(temp_keypoints[:, 1]) + 100, image.shape[0])
                y_start = max(min(temp_keypoints[:, 0]) - 100, 0)
                y_end = min(max(temp_keypoints[:, 0]) + 100, image.shape[1])

                x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)

                person_keypoints[:, 0] = person_keypoints[:, 0] - y_start
                person_keypoints[:, 1] = person_keypoints[:, 1] - x_start

                crop_image = image[x_start:x_end, y_start:y_end, :]
                height, width = crop_image.shape[0], crop_image.shape[1]
                person_keypoints[:, 0] = person_keypoints[:, 0] * 368.0 / width
                person_keypoints[:, 1] = person_keypoints[:, 1] * 368.0 / height

                crop_image = cv2.resize(crop_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                crop_heatmap = create_heatmap(crop_image, person_keypoints)
                x_batch.append(crop_image)
                y_batch.append(crop_heatmap)
            x_batch = np.array(x_batch) / 255.0
            y_batch = np.array(y_batch)
            yield x_batch, y_batch


# if __name__ == '__main__':
#     print(filenames.shape)
#     print(keypoints.shape)
#     print(keypoint_ids_per_image.shape)
#     train_generator()
