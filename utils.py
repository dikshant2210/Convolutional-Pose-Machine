import numpy as np


heatmap_pixel_width = 10


def gaussian_image_map(img_height, img_width, c_y, c_x, variance):
    image_map = np.zeros((img_height, img_width))
    for x in range(c_x - heatmap_pixel_width, c_x + heatmap_pixel_width):
        for y in range(c_y - heatmap_pixel_width, c_y + heatmap_pixel_width):
            dist_sq = (x - c_x) ** 2 + (y - c_y) ** 2
            exponent = dist_sq / (2.0 * variance * variance)
            try:
                image_map[x, y] = np.exp(-exponent)
            except IndexError:
                print(x, y)
    return image_map
