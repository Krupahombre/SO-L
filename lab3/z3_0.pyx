import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_zad3(np.ndarray[np.int32_t, ndim=2] image,
                   np.ndarray[np.int32_t, ndim=2] kernel):
    cdef int width = image.shape[1]
    cdef int height = image.shape[0]
    cdef int k_width = kernel.shape[1]
    cdef int k_height = kernel.shape[0]
    cdef int pad_w = k_width // 2
    cdef int pad_h = k_height // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    cdef np.ndarray[np.int32_t, ndim=2] output = np.zeros_like(image, dtype=np.int32)

    cdef int i, j, ki, kj
    cdef int pixel_value
    for i in range(height):
        for j in range(width):
            pixel_value = 0
            for ki in range(k_height):
                for kj in range(k_width):
                    pixel_value += padded_image[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = pixel_value

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_zad3_parallel(np.ndarray[np.int32_t, ndim=2] image,
                   np.ndarray[np.int32_t, ndim=2] kernel):
    cdef int width = image.shape[1]
    cdef int height = image.shape[0]
    cdef int k_width = kernel.shape[1]
    cdef int k_height = kernel.shape[0]
    cdef int pad_w = k_width // 2
    cdef int pad_h = k_height // 2

    cdef np.ndarray[np.int32_t, ndim=2] padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    cdef np.ndarray[np.int32_t, ndim=2] output = np.zeros_like(image, dtype=np.int32)

    cdef long [:, :] image_save = image
    cdef long [:, :] kernel_save = kernel
    cdef long [:, :] output_save = output

    cdef int i, j, ki, kj
    cdef int pixel_value
    for i in prange(height, nogil=True):
        for j in prange(width):
            pixel_value = 0
            for ki in prange(k_height):
                for kj in prange(k_width):
                    pixel_value += padded_image[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = pixel_value

    return output