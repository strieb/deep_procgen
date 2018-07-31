import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import tensorflow as tf
 
mat = np.random.rand(32, 32)

fil = [[0.7,1,0.7],[1,0,1],[0.7,1,0.7]]
fil = np.reshape(fil,(3,3,1,1,1))

learn = tf.get_variable("learn", (32,32,2), initializer=tf.random_normal_initializer())
soft = tf.nn.softmax(learn)

layer = tf.reshape(soft,(1,32,32,2,1))

conv = tf.nn.conv3d(layer,fil, (1,1,1,1,1), "SAME")
nbr_field = tf.reshape((layer * conv),(32,32,2))
nbr_bonus = tf.reduce_sum(nbr_field,(0,1))
sums = tf.reduce_sum(soft,(0,1))
bonus = tf.reduce_sum((soft[:,:,0] * nbr_field[:,:,0]) ,(0,1))
print(bonus.get_shape())
loss = tf.negative(tf.square(sums[0] - 100)) + bonus - 100

optimizer = tf.train.AdagradOptimizer(learning_rate=1).minimize(-loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for k in range(0, 30):
        for i in range(0, 100):
            l,_ = sess.run((loss,optimizer), feed_dict={})
        print(l)

    l,n = sess.run((bonus, soft), feed_dict={})

print(l)
plt.imshow(n[:,:,0])
plt.show()
