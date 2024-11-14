import time

import matplotlib.pyplot as plt
import numpy as np


lenna = plt.imread("lenna.png")
kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

kernel_size = kernel.shape[0]
conv_result = np.zeros(np.shape(lenna))

start_time = time.time()
lenna_padded = np.pad(lenna, ((1, 1), (1, 1)), mode='constant')

for i in range(lenna.shape[0]):
    for j in range(lenna.shape[1]):
        fragment = lenna_padded[i:i + kernel_size, j:j + kernel_size]
        conv_result[i, j] = np.sum(fragment * kernel)

end_time = time.time()

plt.imshow(conv_result, cmap='gray')
plt.savefig('Z2-0.png')

print(f'Time elapsed: {end_time - start_time}')
