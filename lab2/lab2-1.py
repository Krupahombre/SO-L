import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time

lenna = plt.imread("lenna.png")
kernel = np.array([(-1, -1, -1), (-1, 8, -1), (-1, -1, -1)])

kernel_size = kernel.shape[0]
conv_result = np.zeros(np.shape(lenna))


def calculate_sum(lenna_padded, idx, jdx):
    return np.sum(lenna_padded[idx:idx + kernel_size, jdx:jdx + kernel_size] * kernel)


def row_convolution(lenna_padded, idx):
    return np.array([calculate_sum(lenna_padded, idx, jdx) for jdx in range(lenna_padded.shape[1] - kernel_size + 1)])


def process_row_convolution(params):
    return row_convolution(*params)


def thread_convolution():
    start_time = time.time()
    lenna_padded = np.pad(lenna, ((1, 1), (1, 1)), mode='constant')

    with ProcessPoolExecutor(max_workers=4) as executor:
        row_data_params = [(lenna_padded, i) for i in range(lenna.shape[0])]
        result = list(executor.map(process_row_convolution, row_data_params))

    for i, row_result in enumerate(result):
        conv_result[i, :] = row_result

    end_time = time.time()
    time_elapsed = end_time - start_time

    plt.imshow(conv_result, cmap='gray')
    plt.savefig('Z2-1.png')

    return time_elapsed


if __name__ == '__main__':
    print(f'Time elapsed: {thread_convolution()}')
