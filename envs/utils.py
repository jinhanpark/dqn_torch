import numpy as np


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])