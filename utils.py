import numpy as np
import cv2


heatmap_pixel_width = 15


def gaussian_image_map(img_height, img_width, c_y, c_x, variance):
    image_map = np.zeros((img_height, img_width))
    for x in range(c_x - heatmap_pixel_width, c_x + heatmap_pixel_width):
        for y in range(c_y - heatmap_pixel_width, c_y + heatmap_pixel_width):
            dist_sq = (x - c_x) ** 2 + (y - c_y) ** 2
            exponent = dist_sq / (2.0 * variance * variance)
            try:
                image_map[x, y] = np.exp(-exponent)
            except IndexError:
                pass
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
