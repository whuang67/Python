"""
Implements feature extraction and other data processing helpers.
"""


import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    
    if process_method == 'default':
        # print(data["image"].max())
        data["image"] = data["image"]/255
        # print(data["image"].max())
        
        for i, img_matrix in enumerate(data["image"]):
            data["image"][i] = filters.laplace(img_matrix, ksize = 11)

        remove_data_mean(data)
        
        flatten = np.zeros(shape = (len(data["image"]), 28*28))
        for i, img_matrix in enumerate(data["image"]):
            flatten[i] = img_matrix.flatten()
        data["image"] = flatten
        # print(data["image"].max())
        
    elif process_method == 'raw':
        data["image"] = data["image"]/255
        remove_data_mean(data)

        flatten = np.zeros(shape = (len(data["image"]), 28*28))
        for i, img_matrix in enumerate(data["image"]):
            flatten[i] = img_matrix.flatten()
        data["image"] = flatten

    elif process_method == 'custom':
        pass
    
    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    image_mean = []
    for img_matrix in data["image"]:
        image_mean.append(img_matrix.mean())

    image_mean = np.array(image_mean)
    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    
    mean = compute_image_mean(data)
    for i, img_matrix in enumerate(data["image"]):
        data["image"][i] = img_matrix - mean[i]
    return data
