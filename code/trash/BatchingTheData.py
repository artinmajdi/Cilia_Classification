from __future__ import print_function
from __future__ import division

import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
# Data = mnist.train.images
# Label = mnist.train.Labels

batch_size = 100

def ReadingData():
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
        train_data, train_label, test_data, test_data_ids, driver_ids = pickle.load(f)

    sze = train_data.shape
    train_data2 = np.zeros([sze[0], sze[1] * sze[2]])

    for i in range(sze[0]):
        a = train_data[i, :, :, :]
        a = (a[:, :, 0]/3 + a[:, :, 1]/3 + a[:, :, 2]/3)
        a = np.reshape(a, [1, sze[1] * sze[2]], 'F')
        train_data2[i, :] = a

    return train_data2, train_data, train_label, test_data, test_data_ids, driver_ids

def BatchesList(num_TrainData,batch_size):

    NumBatches = int(num_TrainData/batch_size)
    List = []
    for ind in range(0,NumBatches+1):
        List = np.append(List,np.array(batch_size)*ind)

    if num_TrainData > batch_size*NumBatches:
        List = np.append(List,np.array(num_TrainData))

    return List

def Batch(Data,Label,List,ind):

    a1 = int(List[ind])
    a2 = int(List[ind+1])
    data   = Data[a1:a2,:]
    labels = Label[a1:a2,:]

    return data, labels


TrainData2, TrainData, TrainLabel, TestData, TestDataIds, driver_ids = ReadingData()


# # a = TrainData2[1,:]
# # b = np.reshape(a,[32,24])
# # plt.figure()
# # plt.imshow(b)
# # plt.show()
#
# num_TrainData = TrainData2.shape[0]
#
# indexes = np.random.permutation(TrainData2.shape[0])
# np.take(TrainData2,indexes,axis = 0,out = TrainData2)
# np.take(TrainLabel,indexes,axis = 0,out = TrainLabel)
#
# BatchesEndPoints = BatchesList(num_TrainData,batch_size)
#
# ind = 3
# data,labels = Batch(TrainData2,TrainLabel,BatchesEndPoints,ind)
# #
# # k = TrainData2[351,:]-data[50,:]
# # print(np.unique(k))
# # print(data.shape)
# # print(labels.shape)
