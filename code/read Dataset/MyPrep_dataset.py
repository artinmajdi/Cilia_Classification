from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile

# from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

def load_image(path):
    im = tifffile.imread(path)
    # im = imread(path)
    im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    # im = np.reshape(im,[-1,WIDTH*HEIGHT],order='F')
    return im

def loadData():

    WIDTH, HEIGHT = 32,32
    NUM_CLASSES = 2
    Directory = '/media/data1/artin/code/Cell/data/CiliaImages/'
    Directory1 = Directory + 'class1/'
    Directory2 = Directory + 'class2/'

    subFolders1 = os.listdir(Directory1)
    subFolders2 = os.listdir(Directory2)


    for j in range(NUM_CLASSES):

        if j == 0:
            path = Directory1
            subFolders = subFolders1
        elif j == 1:
            path = Directory2
            subFolders = subFolders2

        Data  = np.zeros((len(subFolders),WIDTH,HEIGHT))
        Label = np.zeros((len(subFolders),1))

        for i in range(len(subFolders)):
            img = load_image( path + subFolders[i] )

            Data[ i , : , : ] = img
            Label[ i , : ] = j

        TestIndexes = np.random.permutation(Data.shape[0])
        K = int(np.floor(0.7*len(TestIndexes)))

        if j == 0:
            A_train = Data[TestIndexes[:K] , : , :]
            L_train = Label[TestIndexes[:K] , :]

            A_test = Data[TestIndexes[K:] , : , :]
            L_test = Label[TestIndexes[K:] , :]

        if j == 1:
            A2_train = Data[TestIndexes[:K] , : , :]
            L2_train = Label[TestIndexes[:K] , :]

            A2_test = Data[TestIndexes[K:] , : , :]
            L2_test = Label[TestIndexes[K:] , :]


    X_train = np.concatenate((A_train , A2_train) , axis = 0)
    Y_train = np.concatenate((L_train , L2_train) , axis = 0)

    TestIndexes = np.random.permutation(X_train.shape[0])
    K = int(np.floor(0.7*len(TestIndexes)))

    X_train = X_train[TestIndexes , : , :]
    Y_train = Y_train[TestIndexes , :]



    X_test = np.concatenate((A_test , A2_test) , axis = 0)
    Y_test = np.concatenate((L_test , L2_test) , axis = 0)

    TestIndexes = np.random.permutation(X_test.shape[0])
    K = int(np.floor(0.7*len(TestIndexes)))

    X_test = X_test[TestIndexes , : , :]
    Y_test = Y_test[TestIndexes , :]



    Y_train = OneHotEncoder(n_values=NUM_CLASSES) \
    .fit_transform(Y_train.reshape(-1, 1)) \
    .toarray()

    Y_test = OneHotEncoder(n_values=NUM_CLASSES) \
    .fit_transform(Y_test.reshape(-1, 1)) \
    .toarray()

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = loadData()
