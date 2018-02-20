from __future__ import print_function
from __future__ import division

import glob
from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as matImage


import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from skimage import color
from scipy.misc import imresize

a = glob.glob("/home/msm/Pictures/*.png")

downsample = 15
WIDTH, HEIGHT = 640 // downsample, 480 // downsample


img = imread((a[0]))
# img = imresize(img, (HEIGHT, WIDTH))
# img = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
#
# img = np.reshape(img,[1,HEIGHT*WIDTH])

#
# img = Image.open(a[0]).convert('L')
# thing = np.asarray(img)
# print(type(img))
# print(np.shape(img))

# plt.figure()
# plt.imshow(img)
# plt.show()

# print(thing.shape)

# thing = np.resize(thing,[80,128])
#
# print(thing.shape)
# plt.imshow(thing[:,:])
# print(thing)
# plt.show()



#
# img = matImage.imread(file)
# img = Image.convert
# img = img.resize((32,32),Image.ANTIALIAS)
#
# print(img.resize)
