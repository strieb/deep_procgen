import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import math
import time


def createInputs(size):
    input = np.zeros((size, size, 2))

    mat = np.random.rand(size, size)-0.5
    mat1 = filters.gaussian_filter(mat, 1) * 1 * 0.6
    mat2 = filters.gaussian_filter(mat, 2) * 2 * 0.6
    mat3 = filters.gaussian_filter(mat, 4) * 4 * 0.6
    mat4 = filters.gaussian_filter(mat, 8) * 8 * 0.7
    mat5 = filters.gaussian_filter(mat, 16) * 16 * 0.7
    mat6 = filters.gaussian_filter(mat, 32) * 32 * 0.8
    mat7 = filters.gaussian_filter(mat, 64) * 64 * 1.0
    mat8 = filters.gaussian_filter(mat, 128) * 128 * 0.6
    mat = mat1+mat2+mat3+mat4+mat5+mat6+mat7+mat8
    heigh = np.zeros((size, size))
    for xi in range(size):
        for yi in range(size):
            x = 2 * xi / size - 1
            y = 2 * yi / size - 1
            heigh[xi, yi] = -x * 0.6
            # heigh[xi, yi] -= max(math.hypot(x, y), 0) - 0.5
    food_mat = (mat*0.4+0.4+heigh).clip(0, 1)
    input[:, :, 0] = food_mat
    return input


def flow(water, height, sand, water_change, volume, sand_change, ax, bx, ay, by, size):
    w1 = water[ay:size-by, ax:size-bx]
    w2 = water[by:size-ay, bx:size-ax]
    h1 = height[ay:size-by, ax:size-bx]
    h2 = height[by:size-ay, bx:size-ax]
    s1 = sand[ay:size-by, ax:size-bx]

    c = np.maximum(np.minimum(w1+h1-(w2+h2), w1), 0) / 4
    s = s1 * c/w1

    water_change[ay:size-by, ax:size-bx] -= c
    water_change[by:size-ay, bx:size-ax] += c
    sand_change[ay:size-by, ax:size-bx] -= s
    sand_change[by:size-ay, bx:size-ax] += s
    volume[ay:size-by, ax:size-bx] += c


def watersim(size, map):
    plt.ion()
    myobj = plt.imshow(map)

    water = np.zeros((size, size))
    sand = np.zeros((size, size))

    series = np.random.rand(20000)-0.5
    series = (filters.gaussian_filter(series, 200) * 30 + 0.5).clip(0.01,1)
    # plt.plot(series)
    # plt.pause(10)
    # series[15000:20000] = 0.01
    # timing = 0
    for i in range(0, 20000):
        water = water + 0.000005 * series[i]
        water[size-1, :] = 0.1
        # water[0, :] = 0.1
        # water[:, size-1] = 0.1
        # water[:, 0] = 0.1
        change = np.zeros((size, size))
        sand_change = np.zeros((size, size))
        volume = np.zeros((size, size))

        # timing -= time.time()
        flow(water, map, sand, change, volume, sand_change, 1, 0, 0, 0, size)
        flow(water, map, sand, change, volume, sand_change, 0, 1, 0, 0, size)
        flow(water, map, sand, change, volume, sand_change, 0, 0, 1, 0, size)
        flow(water, map, sand, change, volume, sand_change, 0, 0, 0, 1, size)
        # timing += time.time()

        erosion = volume*volume/water * 0.04
        sand += erosion + sand_change
        map -= erosion
        map += sand * 0.01
        sand *= 0.99
        water += change
        
        if i % 30 == 0:
            print(series[i])
            # print(timing)
            # timing = 0
            img = np.zeros((size, size, 3))
            w = np.power((water * 20).clip(0, 1),1/3)
            s = (erosion * 10000).clip(0, 1)
            r = (map * 2).clip(0, 1)
            g = (-map * 2+2).clip(0, 1)
            img[:, :, 0] = r*(1-w)+s
            img[:, :, 1] = g*(1-w)+s
            img[:, :, 2] = w+s
            img = img.clip(0, 1)
            # cv2.imshow('image',img)
            # cv2.waitKey(100)
            myobj.set_data(map+water)
            plt.draw()
            plt.pause(0.02)


if __name__ == '__main__':
    plt.ioff()
    demo = createInputs(256)
    watersim(256, demo[:, :, 0])
    plt.imshow(demo[:, :, 0])
    plt.show()









  # flow2(water,map,None,change,volume,None,1,0)
        # flow2(water,map,None,change,volume,None,0,1)
        # flow2(water,map,None,change,volume,None,-1,0)
        # flow2(water,map,None,change,volume,None,0,-1)

        # h1x = map[0:size-1, :]
        # h2x = map[1:size, :]
        # w1x = water[0:size-1, :]
        # w2x = water[1:size, :]
        # changex = np.maximum(np.minimum(w1x+h1x-w2x-h2x, w1x), -w2x) / 4
        # absx = np.abs(changex)

        # change[0:size-1, :] -= changex
        # change[1:size, :] += changex
        # volume[0:size-1, :] += absx
        # volume[1:size, :] += absx

        # h1y = map[:, 0:size-1]
        # h2y = map[:, 1:size]
        # w1y = water[:, 0:size-1]
        # w2y = water[:, 1:size]

        # changey = np.maximum(np.minimum(w1y+h1y-w2y-h2y, w1y), -w2y) / 4
        # change[:, 0:size-1] -= changey
        # change[:, 1:size] += changey
        # volume[:, 0:size-1] += np.abs(changey)
        # volume[:, 1:size] += np.abs(changey)
        
        # for xi in range(0, size-1):
        #     for yi in range(0, size):
        #         f = flow(water[xi, yi], water[xi+1, yi], map[xi, yi], map[xi+1, yi])
        #         change[xi, yi] -= f
        #         volume[xi, yi] += abs(f)
        #         change[xi+ 1, yi] += f
        #         volume[xi+ 1, yi] += abs(f)

        # for xi in range(0, size):
        #     for yi in range(0, size-1):
        #         f = flow(water[xi, yi], water[xi, yi+1], map[xi, yi], map[xi, yi+1])
        #         change[xi, yi] -= f
        #         volume[xi, yi] += abs(f)
        #         change[xi, yi+ 1] += f
        #         volume[xi, yi+1] += abs(f)