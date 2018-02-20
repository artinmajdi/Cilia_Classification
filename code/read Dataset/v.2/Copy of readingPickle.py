
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


# a = unpickle("/media/msm/885E37B75E379D3E/documents/UofA/ECE_523 _ML/project/dataset/data_20.pkl")
# print(a)

from __future__ import print_function
from __future__ import division

import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from skimage.color import rgb2gray

from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
# batch_x, batch_y = mnist.train.next_batch(100)
# b = mnist.test.images

###########################
flags = tf.app.flags
FLAGS = flags.FLAGS

directory = '/media/msm/885E37B75E379D3E/documents/UofA/ECE_523 _ML/project/dataset/data_20.pkl'

flags.DEFINE_boolean('test', False, 'If true, test locally.')

DATASET_PATH = os.environ.get('DATASET_PATH', directory if not FLAGS.test else 'dataset/data_20_subset.pkl')
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')

NUM_EPOCHS = 20 if not FLAGS.test else 1
MAX_FOLDS = 8
BATCH_SIZE = 50

print('Loading dataset {}...'.format(DATASET_PATH))
with open(DATASET_PATH, 'rb') as f:
    # X_train_raw, y_train_raw, driver_ids = pickle.load(f)
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)

# with open(DATASET_PATH, 'wb') as f:
#     pickle.dump((X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids), f,protocol=2)



#


Sz = X_train_raw.shape
b = np.zeros([Sz[0],Sz[1]*Sz[2]])

for i in range(Sz[0]):
    a = X_train_raw[i,:,:,:]
    a = (0.33*a[:,:,0]+0.33*a[:,:,1]+0.33*a[:,:,2])
    a = np.reshape(a,[1,-1],'F')
    b[i,:] = a
#
# a = b[1,:]
# b = np.reshape(a,[32,24])
# b = X_train_raw[1,:,:,:]
# plt.figure()
# plt.imshow(b)
# plt.show()


print(np.shape(b))
print(np.shape(X_train_raw))
print(np.shape(y_train_raw))
print(type(X_train_raw))
print('--------')
# # print(np.shape(batch_x))
# # print(np.shape(batch_y))
# # print(np.shape(b))
# # print(type(b))
# # print(batch_y[1,:])
