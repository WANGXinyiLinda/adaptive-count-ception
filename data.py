'''
data processing.
Assume squared shape inputs.
'''

import pickle
import numpy as np
from skimage.io import imread
from skimage.measure import block_reduce
import scipy.misc
import scipy.stats
import os
from utilities import *

def cal_ef(kernel = "sq", stride = 1, patch_size = 32):
    if (kernel == "sq"):
        ef = (patch_size/stride)**2
    elif (kernel == "gaus"):
        ef = 1.0
    return ef

# density map
def genGausImage(framesize, mx, my, cov):
    x, y = np.mgrid[0:framesize, 0:framesize]
    pos = np.dstack((x, y))
    mean = [mx, my]
    cov = [[cov, 0], [0, cov]]
    rv = scipy.stats.multivariate_normal(mean, cov).pdf(pos)
    return rv/rv.sum()

def getDensity(width, markers, cov, patch_size = 32):
    gaus_img = np.zeros((width,width))
    for k in range(width):
        for l in range(width):
            if (markers[k+patch_size//2,l+patch_size//2] > 0.5):
                gaus_img += genGausImage(width, k, l, cov)
    return gaus_img

# redundant count map
def getMarkersCells(labelPath, base_x = 0 , base_y = 0, scale = 1, patch_size = 32, framesize = 256):
    try:
        lab = imread(labelPath)[:, :, 0]/255 # assum red/white dots with black background
    except:
        lab = imread(labelPath)[:, :]/255 # if only 1 channel
    int_lab = np.zeros(lab.shape)
    int_lab[lab>0.5] = 1
    if scale > 1:
        int_lab = block_reduce(int_lab, block_size=(scale, scale), func=np.max)
    markers = np.pad(int_lab[base_y:base_y+framesize, base_x:base_x+framesize], patch_size, "constant")
    return markers

def getCellCountCells(markers, x_y_h_w = None, noutputs = 1):
    types = [0] * noutputs
    if x_y_h_w != None:
        x, y, h, w = x_y_h_w
        types[0] = markers[y:y+w, x:x+h].sum()
    else:
        types[0] = markers.sum()
    return types

def getLabelsCells(markers, framesize = 256, stride = 1, cov = 4, patch_size = 32, noutputs = 1, kernel = "sq"):
    width = (framesize + patch_size)//stride
    labels = np.zeros((noutputs, width, width))

    if kernel == "sq":
        for x in range(0, width):
            for y in range(0, width):
                count = getCellCountCells(markers,(x*stride, y*stride, patch_size, patch_size), noutputs)
                for i in range(0, noutputs):
                    labels[i][y][x] = count[i]
    elif kernel == "gaus":
        for i in range(0,noutputs):
            labels[i] = getDensity(width, markers, cov, patch_size)
    else:
        print("please choose between gaus/sq for the -kernel argument")

    count_total = getCellCountCells(markers)
    return labels, count_total


def getTrainingExampleCells(img_raw, labelPath, base_x, base_y, scale = 1, stride = 1, patch_size = 32, framesize = 256, kernel = "sq"):
    img = img_raw[base_y:base_y+framesize, base_x:base_x+framesize]
    markers = getMarkersCells(labelPath, base_x, base_y, scale, patch_size, framesize)
    labels, count = getLabelsCells(markers, framesize = framesize, stride = stride, patch_size = patch_size, kernel = kernel)
    return img, labels, count

def data_process(datasetfilename, img_file_path, scale = 1, framesize = 256, slice_stride = 128, kernel = "sq",
                slicing = False, verbose = False):
    if os.path.isfile(datasetfilename):
        print("reading from preprocessed data: ", datasetfilename)
        dataset = pickle.load(open(datasetfilename, "rb" ))
    else:
        dataset = []
        print("total number of images: ", len(img_file_path))
        for path in img_file_path:

            imgPath = path[0]
            im = imread(imgPath)
            img_raw_raw = im.mean(axis=(2))  # grayscale
            img_raw = scipy.misc.imresize(img_raw_raw,
                                        (img_raw_raw.shape[0]//scale, img_raw_raw.shape[1]//scale))
            if verbose:
                print("input image raw shape", img_raw_raw.shape, " ->>>>", img_raw.shape)

            labelPath = path[1]
            if slicing: s = slice_stride
            else: s = framesize
                
            for base_x in range(0, img_raw.shape[0], s):
                for base_y in range(0, img_raw.shape[1], s):
                    if base_x + framesize > img_raw.shape[0] or base_y + framesize > img_raw.shape[1]:
                        continue
                    img, lab, count = getTrainingExampleCells(img_raw, labelPath, base_y, base_x, scale = scale, kernel = kernel)
                    ef = cal_ef(kernel)
                    lab_est = [(l.sum()/ef) for l in lab] #.astype(np.int)
                    if verbose:
                        print("count ", count, "== lab_est", lab_est)
                        print((base_x, base_y))
                    if count != lab_est:
                        print("count ", count, "!= lab_est", lab_est)
                    dataset.append((img, lab, count))

        print("save data to a binary file: ", datasetfilename)
        out = open(datasetfilename, "wb", 0)
        pickle.dump(dataset, out)
        out.close()
    np_dataset = np.asarray(dataset) # shape: (num_data, 3, ?)
    np_dataset = np.rollaxis(np_dataset,1,0) # shape: (3, num_data, ?)
    np_dataset_x = np.asarray([np.transpose(np.asarray([n]), (1,2,0)) for n in np_dataset[0]]) # roll the first axis to last
    np_dataset_y = np.asarray([np.transpose(n, (1,2,0)) for n in np_dataset[1]])
    np_dataset_c = np.asarray([n for n in np_dataset[2]])
    print("image data shape: ", np_dataset_x.shape) 
    print("ground truth data shape", np_dataset_y.shape)
    print("count data shape", np_dataset_c.shape)
    return np_dataset_x, np_dataset_y, np_dataset_c