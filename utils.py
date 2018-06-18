import numpy as np
import cv2
import matplotlib.pyplot as plt


IMAGE_HEIGHT = 368
IMAGE_WIDTH = 368
NUM_CHANNELS = 3
heatmap_pixel_width = 4
decay_rate = 0.5
decay_steps = 20000
init_lr = 0.001
BATCH_SIZE = 16
INTENSITY_THRESHOLD = 0.5
EPOCHS = 15


def gaussian_image_map(img_height, img_width, c_y, c_x, variance):
    image_map = np.zeros((img_height, img_width))
    if c_x < 0 or c_y < 0:
        return image_map
    for x in range(img_width):
        for y in range(img_height):
            dist_sq = (x - c_x) ** 2 + (y - c_y) ** 2
            exponent = dist_sq / (2.0 * variance * variance)
            try:
                image_map[x, y] = np.exp(-exponent)
            except IndexError:
                pass
    # image_map = image_map > INTENSITY_THRESHOLD
    # image_map = image_map.astype(np.float32)
    return image_map


def plot_heatmap(image, heatmap):
    heatmap = np.max(heatmap, axis=-1)
    heatmap = (np.expand_dims(heatmap, axis=-1) * 255.0).astype(np.uint8)
    heatmap_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    final_image = cv2.addWeighted(heatmap_image, 0.5, image, 0.5, 0)
    cv2.imshow('final', final_image)
    cv2.imshow('input', image)
    cv2.imshow('heatmap', heatmap)
    cv2.waitKey(0)


def grid_plot(array):
    plt.figure(figsize=(8, 8))
    for i in range(1, array.shape[-1] + 1):
        plt.subplot(4, 4, i)
        plt.imshow(array[:, :, i - 1], cmap='gray', vmin=0.0, vmax=1.0)
    plt.subplots_adjust(top=.95, bottom=0.05, left=0.10, right=0.95, hspace=0.5, wspace=0.5)
    plt.show()
