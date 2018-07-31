import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import math
import cv2
import time
import colorcet as cc

if __name__ == '__main__':
    rainbow = cc.m_rainbow
    size = 256
    height = np.zeros((size, size))
    for xi in range(size):
        for yi in range(size):
            x = 2 * xi / size - 1
            y = 2 * yi / size - 1
            height[xi, yi] = y * 0.5 + 0.5
    # img = np.zeros((size, size, 4))
    img = rainbow(height)
    # for xi in range(size):
    #     for yi in range(size):
    #         img[xi,yi] = rainbow(height[xi, yi])
    # img = img.clip(0, 255)
    plt.imshow(img)
    plt.show()
