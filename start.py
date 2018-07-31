import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf


def createInputs(n):
    input = np.zeros((n, 16, 16, 1))
    for i in range(0, n):
        mat = np.random.rand(16, 16)-0.5
        mat2 = filters.gaussian_filter(mat, 1)
        mat3 = filters.gaussian_filter(mat, 2)
        # mat4 = filters.gaussian_filter(mat, 4)
        # mat5 = filters.gaussian_filter(mat, 8)
        # mat = mat+mat2*4+mat3*16+mat4*32+mat5*64
        mat = mat+mat2*4+mat3*16
        food_mat = (mat/2 + 0.2).clip(0, 1)
        input[i, :, :, 0] = food_mat

    return input


LAYERS = 4

fertility = tf.placeholder(tf.float32, (None, 16, 16, 1))
fertility_conv = tf.layers.conv2d(fertility, 16, (1, 1), activation=tf.nn.relu,padding="SAME")
fertility_conv = tf.layers.conv2d(fertility_conv, 16, (1, 1), activation=tf.nn.relu,padding="SAME")

fertility_mean = tf.reduce_mean(fertility_conv,(1,2))

rand = tf.placeholder(tf.float32, (None, 16))

rand_fert = tf.concat([rand,fertility_mean],-1)

layer_fc = tf.layers.dense(inputs=(rand_fert), units = 16*16*16, activation=tf.nn.relu)
layer_reshape = tf.reshape((layer_fc), (-1,16,16,16))
layer_both = tf.concat([fertility_conv,layer_reshape],-1)
layer1 = tf.layers.conv2d(layer_both, 16, (3, 3), activation=tf.nn.relu,padding="SAME")
layer2 = tf.layers.conv2d(layer1, 16, (3, 3), activation=tf.nn.relu,padding="SAME")
layer3 = tf.layers.conv2d(layer2, 16, (3, 3), activation=tf.nn.relu,padding="SAME")
layer4 = tf.layers.conv2d(layer3, 32, (1, 1), activation=tf.nn.relu,padding="SAME")
layer5 = tf.layers.conv2d(layer4, LAYERS, (1, 1), activation=None)

buildings = tf.nn.softmax(layer5)
buildings_reshaped = tf.reshape(buildings,(-1,16,16,LAYERS,1))

mask = np.zeros((16,16))
mask[:,:] = 1
mask[1:15,1:15] = 0
mask = mask.reshape(1,16,16)

fil = np.asarray([[0.7,1,0.7],[1,0,1],[0.7,1,0.7]]) / 6.8
fil = np.reshape(fil,(3,3,1,1,1))
buildings_blurred = tf.reshape(tf.nn.conv3d(buildings_reshaped, fil, (1,1,1,1,1), "SAME"),(-1,16,16,LAYERS))
outside = buildings[...,3] + buildings[...,1] + buildings[...,2]
no_wall = tf.reduce_sum(buildings_blurred[...,0] * outside,(1,2))
overfarming = tf.reduce_sum(buildings_blurred[...,1] * buildings[...,1])
bounds = tf.reduce_sum(buildings[...,0] * mask,(1,2))
# nbr_field = tf.reshape((buildings_reshaped * buildings_blurred),(-1,16,16,3))
# nbr_bonus = tf.reduce_sum(nbr_field,(1,2))

buildings_entropy = tf.reduce_sum( tf.square(buildings), (1,2,3))
sums = tf.reduce_sum(buildings, (1, 2))
farms = sums[:, 1]
workers = sums[:, 0]
water = sums[:, 2]
actualFood = tf.minimum((1 + buildings_blurred[..., 2]) * fertility[..., 0], 1)
food = tf.reduce_sum(actualFood * buildings[..., 1] , (1, 2)) * 1
hunger = tf.minimum((food / workers), 1) - 0.5
value =  hunger * workers - farms * 0.4 + tf.sqrt(workers) * 5 - no_wall * 0.2 - water * 0.05 - bounds
# value = -tf.square(workers) * 0.01 - farms * 0.05 + tf.sqrt(workers) * 20 + nbr_bonus[:,0] * 2
loss = tf.negative(tf.reduce_mean(value))

optimizer = tf.train.AdagradOptimizer(learning_rate=0.05).minimize(loss)

init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

plt.ion()

demo = createInputs(1)
demo_ran = np.random.rand(1, 16)
plt.imshow(demo[0, :, :, 0])
plt.pause(5)
plt.draw()

BATCH_SIZE = 1
with tf.Session(config=config) as sess:
    sess.run(init_op)
    l = 0
    for i in range(0, int(20001/BATCH_SIZE)):
        fert = createInputs(BATCH_SIZE)
        ran = np.random.rand(BATCH_SIZE, 16)
        v, _ = sess.run((loss, optimizer), feed_dict={fertility: fert, rand: ran})
        l += v
        if i% (int(500/BATCH_SIZE)) == 0:
            print("value: "+str(l / int(500/BATCH_SIZE)))
            l = 0 
            b, w, f, v = sess.run((buildings, workers, food, value),
                                feed_dict={fertility: demo, rand: demo_ran})

            # fig = plt.figure(figsize=(8, 8))
            # fig.add_subplot(2, 2, 1)
            # plt.imshow(fert[0, :, :, 0])
            # fig.add_subplot(2, 2, 2)
            plt.imshow(b[0][:,:,0:3])
            plt.pause(0.001)
            plt.draw()
            # print("value: "+str(preprocess_input))    

plt.pause(10)
plt.show()