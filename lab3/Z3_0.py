import time
import numpy as np
import matplotlib.pyplot as plt
import z3_0

fig, ax = plt.subplots(1, 2)

kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.int32)

image = plt.imread('lenna.png')
image -= np.min(image)
image /= np.max(image)
image = (image * 255).astype(np.int32)

start_time = time.time()
output = z3_0.convolve_zad3(image, kernel)
output = output[1:image.shape[0], 1:image.shape[1]]
end_time = time.time()

start_time_parallel = time.time()
output_parallel = z3_0.convolve_zad3_parallel(image, kernel)
output_parallel = output_parallel[1:image.shape[0], 1:image.shape[1]]
end_time_parallel = time.time()

print(f'\nKonwolucja - czas wykonania: {end_time - start_time:.6f}')
print(f'\nKonwolucja zrównoleglona - czas wykonania: {end_time_parallel - start_time_parallel:.6f}')

ax[0].imshow(output, cmap='binary_r')
ax[0].set_title(f'Konwolucja')

ax[1].imshow(output_parallel, cmap='binary_r')
ax[1].set_title(f'Konwolucja zrównoleglona')

plt.show()
