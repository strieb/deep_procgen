import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf
import math
from tensorflow.contrib.stateless import stateless_random_uniform
from worldload import Generator

class Procgen:
    size = 23
    inputs_size = 3
    building_size = 3

    def create(self):
        size = self.size  # 23
        size_1 = math.floor(size / 2)  # 11
        size_2 = math.floor(size_1 / 2)  # 5

        seed = tf.placeholder(tf.int64, shape=())

        map = tf.placeholder(tf.float32, (None, size, size, self.inputs_size))
        mask = tf.placeholder(tf.float32, (None, size, size, self.building_size))
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

        layer_0_fc = tf.layers.conv2d(layer_0_c, self.building_size, (1, 1), activation=None)
        buildings_unmasked = tf.nn.softmax(layer_0_fc)

        buildings = buildings_unmasked * mask
        free =1 - tf.reduce_sum(buildings,(3)) + buildings[...,2]
        print(free.get_shape())
        buildings_reshaped = tf.reshape(buildings, (-1, size, size, self.building_size, 1))

        fil = np.asarray([[0.7, 1, 0.7], [1, 0, 1], [0.7, 1, 0.7]]) / 6.8
        fil = np.reshape(fil, (3, 3, 1, 1, 1))
        buildings_blurred = tf.reshape(tf.nn.conv3d(buildings_reshaped, fil, (1, 1, 1, 1, 1), "SAME"), (-1, size, size, self.building_size))

        outside = (free + buildings[...,1]) * (1 - map[...,2])
        no_wall = tf.reduce_sum(buildings_blurred[..., 0] * outside, (1, 2))
        # overfarming = tf.reduce_sum(buildings_blurred[..., 1] * buildings[..., 1])
        hills = tf.reduce_sum(buildings[..., 0] * map[...,1], (1, 2))
        # nbr_field = tf.reshape((buildings_reshaped * buildings_blurred),(-1,16,16,3))
        # nbr_bonus = tf.reduce_sum(nbr_field,(1,2))
        # buildings_entropy = tf.reduce_sum(tf.square(buildings), (1, 2, 3))
        sums = tf.reduce_sum(buildings, (1, 2))
        farms = sums[:, 1]
        workers = sums[:, 0]
        food = tf.reduce_sum(buildings[..., 1] * (map[..., 0] ), (1, 2)) * 0.3
        hunger = tf.minimum((food /(workers+0.01)), 1) - 0.5 #NaN protection
        value = hunger * workers * 2 - farms * 0.1 + tf.sqrt(workers)*3 - no_wall * 1- hills * 1
        # value = -tf.square(workers) * 0.01 - farms * 0.05 + tf.sqrt(workers) * 20 + nbr_bonus[:,0] * 2
        loss = tf.negative(tf.reduce_mean(value))

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

        init_op = tf.global_variables_initializer()

        self.seed = seed
        self.map = map
        self.mask = mask
        self.buildings = buildings
        self.loss = loss
        self.optimizer = optimizer
        self.init_op = init_op
        self.test = workers

def process(img):
    input = np.zeros((img.shape[0],img.shape[1],3))
    mask = np.ones((img.shape[0],img.shape[1],3))

    input[...,0] = img[...,1] #Food
    input[...,1] += (img[...,0]).clip(0,1) #Hills
    input[...,2] = img[...,2] #Water

    
    mask[ :, :, 0] = 0 #Border
    mask[1:img.shape[0]-1, 1:img.shape[1]-1, 0] = 1
    mask[...,1] -= img[...,2] #Water
    mask[...,0] -= img[...,2] #Water

    return np.expand_dims(input.clip(0,1),0), np.expand_dims(mask.clip(0,1),0)

def toImg(processed,buildings):
    p = processed[0]
    b = buildings[0]
    img = np.zeros((p.shape[0], p.shape[1],3))
    img[ ..., 0] = b[..., 0]
    img[..., 1] = b[..., 1]
    img[..., 2] = p[ ..., 2]
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
            if i% (int(500/BATCH_SIZE)) == 0:
                print("value: "+str(l / int(500/BATCH_SIZE)))
                l = 0 
                img = generator.random(23)
                map, mask = process(img)
                b = sess.run((procgen.buildings, procgen.test), feed_dict={procgen.map: map, procgen.mask: mask, procgen.seed: 0})
                
                plt.imshow(img)
                plt.pause(1)
                plt.imshow(toImg(map,b))
                plt.pause(0.001)
                plt.draw()
                # print("value: "+str(preprocess_input))    

    plt.pause(10)
    plt.show()