import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import math
import cv2
import time
import colorcet as cc

def createInputs(size):

    mat = np.random.rand(size, size)-0.5
    mat1 = filters.gaussian_filter(mat, 1) * 1 * 0.8
    mat2 = filters.gaussian_filter(mat, 2) * 2 * 0.8
    mat3 = filters.gaussian_filter(mat, 4) * 4 * 1.0
    mat4 = filters.gaussian_filter(mat, 8) * 8 * 2.2
    mat5 = filters.gaussian_filter(mat, 16) * 16 * 2.2
    mat6 = filters.gaussian_filter(mat, 32) * 32 * 2.2
    mat7 = filters.gaussian_filter(mat, 64) * 64 * 0.8
    mat8 = filters.gaussian_filter(mat, 128) * 128 * 0.6
    mat = mat1+mat2+mat3+mat4+mat5+mat6+mat7+mat8
    height = np.zeros((size, size))
    for xi in range(size):
        for yi in range(size):
            x = 2 * xi / size - 1
            y = 2 * yi / size - 1
            height[xi, yi] = -x * 2
            # height[xi, yi] -= max(math.hypot(x, y), 0) * 0.5 - 0.4
    return (mat*0.8+1.5+height).clip(-0.2, 4)


def flow(water,height, outflow, dir, size):
    ax = 0
    bx = 0
    ay = 0
    by = 0
    if dir == 0:
        ax=1
        dir_neg = 1
    if dir == 1:
        bx=1
        dir_neg = 0
    if dir == 2:
        ay=1
        dir_neg = 3
    if dir == 3:
        by=1
        dir_neg = 2

    w1 = water[ay:size-by, ax:size-bx]
    w2 = water[by:size-ay, bx:size-ax]
    h1 = height[ay:size-by, ax:size-bx]
    h2 = height[by:size-ay, bx:size-ax]

    c = np.maximum(np.minimum(w1+h1-(w2+h2), w1), 0) / 5

    outflow[dir, ay:size-by, ax:size-bx] += c
    outflow[dir_neg, by:size-ay, bx:size-ax] -= c

def energy_flow(water,height, outflow, energy, dir, size):
    ax = 0
    bx = 0
    ay = 0
    by = 0
    if dir == 0:
        ax=1
        dir_neg = 1
    if dir == 1:
        bx=1
        dir_neg = 0
    if dir == 2:
        ay=1
        dir_neg = 3
    if dir == 3:
        by=1
        dir_neg = 2

    w1 = water[ay:size-by, ax:size-bx]
    w2 = water[by:size-ay, bx:size-ax]
    h1 = height[ay:size-by, ax:size-bx]
    h2 = height[by:size-ay, bx:size-ax]
    
    o = outflow[dir, ay:size-by, ax:size-bx]
    e = np.maximum(w1+h1-(w2+h2), 0) * o

    energy[ay:size-by, ax:size-bx] += e
    energy[by:size-ay, bx:size-ax] += 0

def damp(factor, outflow, dir, size):
    ax = 0
    bx = 0
    ay = 0
    by = 0
    if dir == 0:
        ax=1
        dir_neg = 1
    if dir == 1:
        bx=1
        dir_neg = 0
    if dir == 2:
        ay=1
        dir_neg = 3
    if dir == 3:
        by=1
        dir_neg = 2

    damp = np.maximum(outflow[dir, ay:size-by, ax:size-bx],0) * factor[ay:size-by, ax:size-bx]

    outflow[dir, ay:size-by, ax:size-bx] -= damp
    outflow[dir_neg, by:size-ay, bx:size-ax] += damp

def sand_flow(out, dillation, sand_change, dir, size):
    ax = 0
    bx = 0
    ay = 0
    by = 0
    if dir == 0:
        ax=1
    if dir == 1:
        bx=1
    if dir == 2:
        ay=1
    if dir == 3:
        by=1

    o = out[dir, ay:size-by, ax:size-bx]
    d = dillation[ay:size-by, ax:size-bx]

    sand_change[ay:size-by, ax:size-bx] -= o * d
    sand_change[by:size-ay, bx:size-ax] += o * d

show_water = True
show_sand = True

def press(event):
    global show_sand
    global show_water
    if event.key == 'w':
        show_water = not show_water
    if event.key == 'e':
        show_sand = not show_sand



def watersim(height, size):
    
    global show_sand
    global show_water

    rainbow = cc.m_rainbow
    water = np.zeros((size, size))
    sand = np.zeros((size, size))
    energy = np.zeros((size, size))
    speed_max = np.zeros((size, size))
    water_long = np.zeros((size, size))
    # max_erosion = filters.gaussian_filter(np.random.rand(size, size), 0.5).clip(0.2,1)

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)

    img = rainbow((height +0.3) /1.3)
    myobj = plt.imshow(img)
    outflow = np.zeros((4, size, size))
    # water[20:40, 40:60] += 2
    # water += 0.5

    series = np.random.rand(10000)-0.5
    series = (filters.gaussian_filter(series, 100) * 30 + 1).clip(0.2,2)
    height[size-5:size,:] = -0.2
    # height[0:5,:] = -0.2
    # height[:,size-5:size] = -0.2
    # height[:,0:5] = -0.2
    for i in range(0, 5000):
        if(i % 1000 == 0):
            print(i)
        if(i < 4000):
            water += 0.000003 * series[i]
        else:
            water += 0.000003
            water_long += water
        if(i < 800):
            water += 0.0001 * (math.sin(i/50)+2)
            water *= 0.99
            # sand *= 0.9
        if(i < 2000):
            water += 0.00001 * (math.sin(i/50)+2)
        water[size-1,:] = 0.2
        # water[0,:] = 0.2
        # water[:,size-1] = 0.2
        # water[:,0] = 0.2
        # timing -= time.time()
        flow(water,height, outflow, 0, size)
        flow(water,height, outflow, 1, size)
        flow(water,height, outflow, 2, size)
        flow(water,height, outflow, 3, size)
        # timing += time.time()
        out = np.maximum(outflow,0)
        volume = np.sum(out,axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            energy.fill(0)
            energy_flow(water,height,out, energy, 0, size)
            energy_flow(water,height,out, energy, 1, size)
            energy_flow(water,height,out, energy, 2, size)
            energy_flow(water,height,out, energy, 3, size)

            speed_max = speed_max * 0.8 + 0.2 * np.nan_to_num(np.minimum(energy / volume * 50 + 0.05, 1),0)
            volume_max = np.minimum( volume,speed_max * water)
            factor = 1 - np.nan_to_num(volume_max / volume,0)

            damp(factor, outflow, 0, size)
            damp(factor, outflow, 1, size)
            damp(factor, outflow, 2, size)
            damp(factor, outflow, 3, size)

            out = np.maximum(outflow,0)
            dillation = np.minimum( np.nan_to_num(sand / water, 0), 1)
            sand_change = np.zeros((size, size))
            
            sand_flow(out, dillation, sand_change, 0, size)
            sand_flow(out, dillation, sand_change, 1, size)
            sand_flow(out, dillation, sand_change, 2, size)
            sand_flow(out, dillation, sand_change, 3, size)

            volume = np.sum(out, axis=0)
            speed = np.nan_to_num(volume / water, 0)
            
            energy.fill(0)
            energy_flow(water,height,out, energy, 0, size)
            energy_flow(water,height,out, energy, 1, size)
            energy_flow(water,height,out, energy, 2, size)
            energy_flow(water,height,out, energy, 3, size)

            erosion =  energy * 4 # * max_erosion
            erosion = filters.gaussian_filter(erosion, 0.3)
            erosion = np.minimum(erosion, 0.1)
            # erosion_min = erosion >= 0.00001
            # erosion *= erosion_min
            # erosion =  volume * speed * 0.03
            sand += erosion + sand_change
            min_water = np.minimum(water * 100,1)
            height -= erosion
            sanding = (1-min_water)* 0.6 +0.2
            height += sand * sanding
            sand *= 1 - sanding

        change = np.sum(outflow,axis=0)
        water -= change
        water = np.maximum(water,0)
        outflow *= 0.98

        if i % 20 == 0:
            # print(series[i])
            # img = np.zeros((size, size, 3))
            img = rainbow((height +0.6) /4)
            w = (water * 1000).clip(0, 1)
            s = (sand*1000).clip(0,1) 
            if(show_water):
                img[:, :, 0] += - w * 0.2
                img[:, :, 1] += - w * 0.2
                img[:, :, 2] += w
            if(show_sand):
                img[:, :, 0] += s
                img[:, :, 1] += s
            # s = w * 0.5
            # s = (sand * 10).clip(0, 1)
            # r = (height * 1.8).clip(0, 1)
            # g = (-height * 1.8 + 1.8).clip(0, 1) + height.clip(-1,0)
            # img[:, :, 0] = r*(1-w) + s
            # img[:, :, 1] = g*(1-w) + s
            img = img.clip(0, 1)
            # cv2.imshow('image',img)
            # cv2.waitKey(100)

            myobj.set_data(img)
            plt.draw()
            plt.pause(0.02)

    info = {
        'water': water_long,
        'height': height
    }
    np.save('C:/ml/worldgen/worlds/'+str(time.time()),info)

if __name__ == '__main__':
    for i in range(0,20):
        height = createInputs(256)
        watersim(height, 256)
