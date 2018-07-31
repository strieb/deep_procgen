import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf
import math
from tensorflow.contrib.stateless import stateless_random_uniform
from worldload import Generator
from create_venice_network import create_value
from create_stride_network import create_network


class Procgen:
    size = 23
    inputs_size = 3
    output_size = 4

    create_network = staticmethod(create_network)
    create_value = staticmethod(create_value)

    def create(self):
        size = self.size

        seed = tf.placeholder(tf.int64, shape=())
        map = tf.placeholder(tf.float32, (None, size, size, self.inputs_size))
        mask = tf.placeholder(tf.float32, (None, size, size, self.output_size))

        outputs_unmasked = self.create_network(size, map, mask, seed, self.output_size)
        outputs_masked = outputs_unmasked * mask
        outputs_free = 1 - tf.reduce_sum(outputs_masked, (3)) + outputs_masked[..., self.output_size-1]

        outputs = tf.concat([outputs_masked[...,:self.output_size-1],tf.expand_dims(outputs_free,-1)],-1)

        value = self.create_value(size,map,outputs,self.output_size)

        loss = tf.negative(tf.reduce_mean(value))

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

        init_op = tf.global_variables_initializer()

        self.seed = seed
        self.map = map
        self.mask = mask
        self.buildings = outputs
        self.loss = loss
        self.optimizer = optimizer
        self.init_op = init_op


def process(img):
    input = np.zeros((img.shape[0], img.shape[1], 3))
    mask = np.ones((img.shape[0], img.shape[1], 4))

    input[..., 0] = img[..., 1]  # Food
    input[..., 1] += (img[..., 0]).clip(0, 1)  # Hills
    input[..., 2] = img[..., 2]  # Water

    mask[:, :, 0] = 0  # Border
    mask[1:img.shape[0]-1, 1:img.shape[1]-1, 0] = 1
    mask[..., 1] -= img[..., 2]  # Water
    mask[..., 0] -= img[..., 2]  # Water

    return np.expand_dims(input.clip(0, 1), 0), np.expand_dims(mask.clip(0, 1), 0)


def toImg(processed, buildings):
    p = processed[0]
    b = buildings[0]
    img = np.zeros((p.shape[0], p.shape[1], 3))
    img[..., 0] = b[..., 0]
    img[..., 1] = b[..., 1]
    img[..., 2] = p[..., 2] * 0.5 + b[..., 2] * 0.5
    return img


if __name__ == '__main__':
    procgen = Procgen()
    generator = Generator()
    procgen.create()

    plt.ion()
    plt.draw()

    BATCH_SIZE = 1

    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(procgen.init_op)
        l = 0
        for i in range(0, int(40001/BATCH_SIZE)):
            map, mask = process(generator.random(23))
            v, _ = sess.run((procgen.loss, procgen.optimizer), feed_dict={procgen.map: map, procgen.mask: mask, procgen.seed: i})
            l += v
            if i % (int(500/BATCH_SIZE)) == 0:
                print("value: "+str(l / int(500/BATCH_SIZE)))
                l = 0
                img = generator.random(23)
                map, mask = process(img)
                b = sess.run((procgen.buildings), feed_dict={procgen.map: map, procgen.mask: mask, procgen.seed: 0})

                plt.imshow(img)
                plt.pause(1)
                plt.imshow(toImg(map, b))
                plt.pause(0.001)
                plt.draw()
                # print("value: "+str(preprocess_input))

    plt.pause(10)
    plt.show()
