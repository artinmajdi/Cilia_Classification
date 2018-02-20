import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

Data = mnist.train.images
Label = mnist.train.Labels

batch_size = 100
num_TrainData = Data.shape[0]

indexes = np.random.permutation(Data.shape[0])
np.take(Data ,indexes,axis = 0,out = Data)
np.take(Label,indexes,axis = 0,out = Label)

def BatchesList(num_TrainData,batch_size):

    NumBatches = int(num_TrainData/batch_size)
    List = []
    for ind in range(0,NumBatches+1):
        List = np.append(List,np.array(batch_size)*ind)

    if num_TrainData > batch_size*NumBatches:
        List = np.append(List,np.array(num_TrainData))

    return List

def Batch(Data,Label,List,ind):

    data   = Data[List(ind):List(ind+1),:]
    labels = Label[List(ind):List(ind+1),:]

    return data, labels


BatchesEndPoints = BatchesList(num_TrainData,batch_size)
print(BatchesEndPoints)

ind = 3
data,labels = Batch(Data,Label,BatchesEndPoints,ind)
