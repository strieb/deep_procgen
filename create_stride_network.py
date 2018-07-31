import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf
import math
from tensorflow.contrib.stateless import stateless_random_uniform
from worldload import Generator


def create_network(size, map, mask, seed, output_size):
    with tf.name_scope("stride_network"):
        size_1 = math.floor(size / 2)  # 11
        size_2 = math.floor(size_1 / 2)  # 5

        in_all = tf.concat([map,mask],-1)

        input_layer_0_b = tf.layers.conv2d(in_all, 8, (3, 3), activation=tf.nn.relu, padding="SAME")

        input_layer_1_a = tf.layers.conv2d(input_layer_0_b, 8, (3, 3), strides=(2, 2), activation=tf.nn.relu, padding="VALID")
        input_layer_1_b = tf.layers.conv2d(input_layer_1_a, 8, (3, 3),  activation=tf.nn.relu, padding="SAME")

        input_layer_2_a = tf.layers.conv2d(input_layer_1_b, 16, (3, 3), strides=(2, 2), activation=tf.nn.relu, padding="VALID")
        input_layer_2_b = tf.layers.conv2d(input_layer_2_a, 16, (3, 3),  activation=tf.nn.relu, padding="SAME")

        input_layer_3_a = tf.layers.conv2d(input_layer_2_b, 32, (size_2, size_2), activation=tf.nn.relu, padding="VALID")

        random_1 = stateless_random_uniform((1, 1, 1, 16), [seed, 0])
        input_layer_3_b = tf.concat([input_layer_3_a, random_1], -1)
        input_layer_3_c = tf.layers.conv2d(input_layer_3_b, 32, (1, 1), activation=tf.nn.relu)

        layer_2_transpose = tf.layers.conv2d_transpose(input_layer_3_c, 16, (size_2, size_2))
        layer_2_conc = tf.concat([layer_2_transpose, input_layer_2_b], -1)
        layer_2_b = tf.layers.conv2d(layer_2_conc, 16, (3, 3),  activation=tf.nn.relu, padding="SAME")

        layer_1_transpose = tf.layers.conv2d_transpose(layer_2_b, 8, (3, 3), strides=(2, 2))
        layer_1_conc = tf.concat([layer_1_transpose, input_layer_1_b], -1)
        layer_1_b = tf.layers.conv2d(layer_1_conc, 16, (3, 3),  activation=tf.nn.relu, padding="SAME")

        layer_0_transpose = tf.layers.conv2d_transpose(layer_1_b, 8, (3, 3), strides=(2, 2))
        layer_0_conc = tf.concat([layer_0_transpose, input_layer_0_b], -1)
        layer_0_b = tf.layers.conv2d(layer_0_conc, 16, (3, 3),  activation=tf.nn.relu, padding="SAME")
        layer_0_c = tf.layers.conv2d(layer_0_b, 32, (1, 1),  activation=tf.nn.relu, padding="SAME")

        layer_0_fc = tf.layers.conv2d(layer_0_c, output_size, (1, 1), activation=None)
        outputs = tf.nn.softmax(layer_0_fc)
        return outputs