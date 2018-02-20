
from __future__ import print_function
from __future__ import division

import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tifffile
from sklearn.preprocessing import OneHotEncoder
batch_size = 128
n_classes = 2
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
fcNumNeurons = 1024


n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

trainSize = 0.7
hm_epochs = 50
downsample = 10

HEIGHT = 32
WIDTH = 32

x = tf.placeholder('float',[None,HEIGHT*WIDTH])
y = tf.placeholder('float')

Directory = '/media/data1/artin/code/Cell_Classification/data/CiliaImages/'

def ConvertingData(train_data):

    train_data = (train_data[:, :, :, 0] / 3 + train_data[:, :, :, 1] / 3  + train_data[:, :, :, 2] / 3 )
    sze = train_data.shape
    train_data2 = np.zeros([sze[0], sze[1]*sze[2]])

    for i in range(sze[0]):
        a = train_data[i, :, :]
    #     a = (a[:, :, 0]/3 + a[:, :, 1]/3 + a[:, :, 2]/3)
        a = np.reshape(a, [1, sze[1] * sze[2]], 'F')
        train_data2[i, :] = a

    return train_data2

def load_image(path):
    im = tifffile.imread(path)
    im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    im = np.reshape(im,[-1,WIDTH*HEIGHT],order='F')
    return im

def ReadingData(trainSize):


    WIDTH, HEIGHT = 32,32
    NUM_CLASSES = 2
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

        Data  = np.zeros((len(subFolders),WIDTH*HEIGHT))
        Label = np.zeros((len(subFolders),1))

        for i in range(len(subFolders)):
            img = load_image( path + subFolders[i] )

            Data[ i , : ] = img
            Label[ i , : ] = j

        TestIndexes = np.random.permutation(Data.shape[0])
        K = int(np.floor(trainSize*len(TestIndexes)))

        if j == 0:
            A_train = Data[TestIndexes[:K] , : ]
            L_train = Label[TestIndexes[:K] , :]

            A_test = Data[TestIndexes[K:] , : ]
            L_test = Label[TestIndexes[K:] , :]

        if j == 1:
            A2_train = Data[TestIndexes[:K] , : ]
            L2_train = Label[TestIndexes[:K] , :]

            A2_test = Data[TestIndexes[K:] , : ]
            L2_test = Label[TestIndexes[K:] , :]


    X_train = np.concatenate((A_train , A2_train) , axis = 0)
    Y_train = np.concatenate((L_train , L2_train) , axis = 0)

    TestIndexes = np.random.permutation(X_train.shape[0])
    K = int(np.floor(0.7*len(TestIndexes)))

    X_train = X_train[TestIndexes , : ]
    Y_train = Y_train[TestIndexes , :]



    X_test = np.concatenate((A_test , A2_test) , axis = 0)
    Y_test = np.concatenate((L_test , L2_test) , axis = 0)

    TestIndexes = np.random.permutation(X_test.shape[0])
    K = int(np.floor(0.7*len(TestIndexes)))

    X_test = X_test[TestIndexes , : ]
    Y_test = Y_test[TestIndexes , :]



    Y_train = OneHotEncoder(n_values=NUM_CLASSES) \
    .fit_transform(Y_train.reshape(-1, 1)) \
    .toarray()

    Y_test = OneHotEncoder(n_values=NUM_CLASSES) \
    .fit_transform(Y_test.reshape(-1, 1)) \
    .toarray()

    return X_train, Y_train, X_test, Y_test

def BatchesList(num_TrainData,batch_size):

    NumBatches = int(num_TrainData/batch_size)
    List = []
    for ind in range(0,NumBatches+1):
        List = np.append(List,np.array(batch_size)*ind)

    if num_TrainData > batch_size*NumBatches:
        List = np.append(List,np.array(num_TrainData-1))

    return List

def Batch(Data,Label,List,ind):

    a1 = int(List[ind])
    a2 = int(List[ind+1])
    data   = Data[a1:a2,:]
    labels = Label[a1:a2,:]

    return data, labels

mode = 'Training' # 'NotTraining'
if mode == 'Training':
    TrainData, TrainLabel, TestData, TestLabel = ReadingData(trainSize)

    num_TrainData = TrainData.shape[0]
    BatchesEndPointsTrain = BatchesList(num_TrainData,batch_size)
    BatchesEndPointsTest = BatchesList(TestData.shape[0],batch_size)


def neural_network_model(data):
    hidden_1_layer = {'Weights':tf.Variable(tf.random_normal([WIDTH*HEIGHT,n_nodes_hl1])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'Weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'Weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                       'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'Weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                       'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['Weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['Weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['Weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['Weights']) + output_layer['biases']
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # this part in the video is (prediction,y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # why it doesn't have num_examples
            for ind in range(len(BatchesEndPointsTrain)-1):
                batch_x, batch_y = Batch(TrainData, TrainLabel, BatchesEndPointsTrain, ind)
                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x , y: batch_y})
                epoch_loss += c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        A = tf.cast(correct,'float')
        accuracy = tf.reduce_mean(A)
        # Accuracy = accuracy.eval({x:TestData , y:TestLabel})

        span = BatchesList(TestData.shape[0],1000)
        Accuracy = np.zeros((TestData.shape[0]))
        for ind in range(len(span)-1):
            batch_x, batch_y = Batch(TestData, TestLabel, span, ind)

            a = int(span[ind])
            b = int(span[ind+1])
            Accuracy[a:b]  = A.eval({x:batch_x , y:batch_y})

        Accuracy = np.mean(Accuracy)
        print('MLP Accuracy:',Accuracy)

    return Accuracy


Accuracy = train_neural_network(x)
