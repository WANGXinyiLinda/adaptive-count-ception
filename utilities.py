'''
helper functions.
assume square-shaped image input.
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import floor
from skimage.transform import rotate
from skimage import exposure

def cal_ef(kernel = "sq", stride = 1, patch_size = 32):
    if (kernel == "sq"):
        ef = (patch_size/stride)**2
    elif (kernel == "gaus"):
        ef = 1.0
    return ef

def sum_count_map(m, kernel = "sq"):
	ef = cal_ef(kernel)   
	return np.asarray([np.sum(p)/ef for p in m])

def plot_map(m, file_path = None):
    # m is like (256, 256, 1)
    a = np.reshape(m, (m.shape[0], m.shape[1]))
    plt.imshow(a)
    if file_path != None:
        plt.savefig(file_path)

def rotate_img(m, degree = 90):
    return rotate(m, degree)

def change_exposure(m, gamma = 0.2):
    return exposure.adjust_gamma(m, gamma) 
    