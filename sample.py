import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from skimage.measure import block_reduce

# mat[30:32,30:40]=1
# mat[33:35,30:40]=-1

def createFilter(n):
    s = n + n - 1
    xx, yy = np.mgrid[:s, :s]
    mat = (n- np.abs(xx-n+1)) * (n-np.abs(yy-n+1)) / (n * n)
    return mat

def createCostMatrix(n):
    ns = n // 2
    xx, yy = np.mgrid[:n, :n]
    mat = np.hypot(xx-ns,yy-ns)
    return mat

def slurp(producer, consumer):
    ones = np.ones((3,3))
    cost_matrix = createCostMatrix(3)

    consumer_spread = filters.convolve(consumer, ones, mode='constant') # spread consumption
    static_consumer = producer - np.minimum(consumer_spread, producer) # don't pread overproduction
    producer_capped = np.where(producer == 0, 0, producer / (consumer_spread + static_consumer)) # Is spread to own and surrounding tiles
    producer_spread = filters.convolve(producer_capped, ones, mode='constant') # spread it
    slurp = producer_spread * (consumer)+static_consumer # consumers slurp it

    cost_spread = filters.convolve(consumer, cost_matrix, mode='constant') # same as consumer spread
    cost = np.sum(producer_capped * cost_spread)
    return slurp, cost

def downsample(mat, n):
    weights = createFilter(n)
    conv = filters.convolve(mat, weights, mode='constant')
    conv = conv[::2,::2]
    return conv

producer = np.zeros((65, 65))
consumer = np.zeros((65, 65))
producer[12:22,12] = 1
consumer[13:23,13] = 1

factor = 1
c_max = 0
for i in range(7):

    sub = np.minimum(producer,consumer)
    consumer -= sub
    producer -= sub

    factor *= 2
    producer = downsample(producer,2)
    consumer = downsample(consumer,2)
    sub = np.minimum(producer,consumer)
    consumer -= sub
    producer -= sub
    plt.imshow(producer- consumer)
    plt.show()
    producer, cost = slurp(producer, consumer)

    c_max += factor * cost
    print(factor * cost)
    plt.imshow(producer- consumer)
    plt.show()
print(c_max)

# weights = createFilter(4)
# for i in range(10):
#     mat = np.zeros((65, 65))
#     mat[0+i,21]=1

#     mat = filters.convolve(mat, weights, mode='constant')
#     mat = mat[::4,::4]
#     print(np.sum(mat))
#     plt.imshow(mat,vmin=0,vmax=1)
#     plt.show()
