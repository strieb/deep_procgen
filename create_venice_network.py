import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf
import math
from tensorflow.contrib.stateless import stateless_random_uniform
from worldload import Generator

def create_value(size, map, buildings, buildings_size):
    
    buildings_reshaped = tf.reshape(buildings, (-1, size, size, buildings_size, 1))

    fil = np.asarray([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]]) / 6.8
    fil = np.reshape(fil, (3, 3, 1, 1, 1))
    buildings_blurred = tf.reshape(tf.nn.conv3d(buildings_reshaped, fil, (1, 1, 1, 1, 1), "SAME"), (-1, size, size, buildings_size))

    outside = (buildings[...,1]+ buildings[...,3])
    no_wall = tf.reduce_sum(buildings_blurred[..., 0] * outside, (1, 2))
    trade =  tf.reduce_sum(tf.tanh((buildings_blurred[..., 2]) * 10) * buildings[..., 0] , (1, 2))
    hills = tf.reduce_sum(buildings[..., 0] * map[...,1], (1, 2))
    sums = tf.reduce_sum(buildings, (1, 2))
    farms = sums[:, 1]
    workers = sums[:, 0]
    water = sums[:, 2]
    food = tf.reduce_sum(buildings[..., 1] * (map[..., 0]), (1, 2)) * 0.3
    hunger = tf.minimum((food /(workers+0.01)), 1) - 0.5 #NaN protection
    value = hunger * workers * 2 - farms * 0.1 - water * 0.1 + tf.sqrt(workers)*6 - no_wall * 1 - hills * 2 + trade * 0.5
    
    return value