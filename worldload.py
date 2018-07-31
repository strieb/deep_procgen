import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import os
import random

class Generator:
    
    border = 10

    def __init__(self):
        directory = 'C:/ml/worldgen/worlds/'
        self.maps = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            world = np.load(directory+file)
            water = world.item().get('water')
            height = world.item().get('height')
            water_filter = (water > 0.1) * 1
            height_filter = np.hypot(filters.sobel(height,0),filters.sobel(height,1))
            fert = np.minimum(filters.gaussian_filter(water,1) * 30 ,1)-height_filter*0.6 + 0.2
            converted = np.zeros((256,256,3))
            converted[...,0] = height_filter * (1-water_filter)
            converted[...,1] = fert * (1-water_filter)
            converted[...,2] = water_filter
            converted = converted.clip(0,1)
            self.maps.append(converted)
            # img[...,0] = height_filter * (1-water_filter) * 0.5
    
    def random(self, size):
        map = random.choice(self.maps)
        y_max = map.shape[0] - size - self.border
        x_max = map.shape[0] - size - self.border
        y = random.randint(self.border, y_max-1)
        x = random.randint(self.border, x_max-1)
        slice = map[y:y+size,x:x+size,:]
        if np.sum(slice[...,2]) >= size * size * 0.95:
            return self.random(size)
        return slice


if __name__ == '__main__':
    generator = Generator()
    for i in range(20):
        plt.imshow(generator.random(32))
        plt.show()