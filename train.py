import tensorflow as tf
import dbcollection as dbc
import cv2
import numpy as np
import os
from model import ConvolutionalPoseMachine as CPM
from utils import decay_steps, decay_rate, init_lr, EPOCHS, gaussian_image_map, \
    IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE, grid_plot


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

data_split_index = int(len(file_index) * 0.8)
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
    for start in range(0, len(file_index[:data_split_index]), BATCH_SIZE):
        end = min(start + BATCH_SIZE, data_split_index)
        x_batch, y_batch = list(), list()

        for index in range(start, end):
            # continue
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
        x_batch = np.array(x_batch) / 255.0 - 0.5
        y_batch = np.array(y_batch)
        yield x_batch, y_batch


def main(argv):
    train_log_save_dir = os.path.join('model', 'logs', 'train')
    test_log_save_dir = os.path.join('model', 'logs', 'test')
    os.system('mkdir -p {}'.format(train_log_save_dir))
    os.system('mkdir -p {}'.format(test_log_save_dir))

    model = CPM(stages=3, joints=16)
    model.build_model()
    model.build_loss(decay_rate=decay_rate, decay_steps=decay_steps, lr=init_lr)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=None)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        num_iterations = int(data_split_index / BATCH_SIZE) + 1

        # path = 'model/weights/model.ckpt-1'
        # saver.restore(sess, path)
        # begin = int(path.split('-')[1]) + 1
        begin = 0

        print('Starting training...')
        for epoch in range(begin, EPOCHS):
            loss = 0
            count = 0
            for start in range(0, len(file_index[:data_split_index]), BATCH_SIZE):
                end = min(start + BATCH_SIZE, data_split_index)
                x_batch, y_batch = list(), list()
                count += 1

                for index in range(start, end):
                    image_path = ascii2str(filenames[file_index[index]])
                    person_keypoints = keypoints[index]
                    temp_keypoints = person_keypoints

                    image = cv2.imread(image_path)

                    temp_keypoints = temp_keypoints[np.where((person_keypoints[:, 2] != -1) *
                                                             (person_keypoints[:, 0] != 0) *
                                                             (person_keypoints[:, 1] != 0))]
                    try:
                        x_start = max(min(temp_keypoints[:, 1]) - 100, 0)
                        x_end = min(max(temp_keypoints[:, 1]) + 100, image.shape[0])
                        y_start = max(min(temp_keypoints[:, 0]) - 100, 0)
                        y_end = min(max(temp_keypoints[:, 0]) + 100, image.shape[1])
                    except ValueError:
                        continue

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
                x_batch = np.array(x_batch) / 255.0 - 0.5
                y_batch = np.array(y_batch)

                stage_loss, total_loss, train_op, summaries, heatmaps, \
                    global_step, learning_rate = sess.run([model.stage_loss,
                                                           model.total_loss,
                                                           model.train_op,
                                                           merged_summary,
                                                           model.heatmaps,
                                                           model.global_step,
                                                           model.learning_rate],
                                                          feed_dict={model.images: x_batch,
                                                                     model.true_heatmaps: y_batch})
                train_writer.add_summary(summaries, global_step)
                loss += total_loss
                print('\tEpoch: {} Iteration: {}/{}, Total Average loss: {}'.format(epoch+1, count, num_iterations,
                                                                                    loss / count))
            print('Epoch {} completed.'.format(epoch + 1))
            saver.save(sess=sess, save_path='model/weights/model.ckpt', global_step=epoch + 1)


if __name__ == '__main__':
    tf.app.run()
