from pyexpat import model
from sklearn.svm import SVC
from scipy.signal import spectrogram
from sklearn.neural_network import MLPClassifier

import glob
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch
import torch.nn as nn

# 引数はCNN_hyper_paramsでまとめる
def CNN_conv1D_keras(filters_list, kernel, act, stride, dropout_list):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=filters_list[0], kernel_size=kernel, activation=act,
                    input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(filters=filters_list[1], kernel_size=kernel, activation=act))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(filters=filters_list[2], kernel_size=kernel, activation=act))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv1D(filters=filters_list[3], kernel_size=kernel, activation=act))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_list[0]))
    model.add(tf.keras.layers.Conv1D(filters=filters_list[4], kernel_size=kernel, activation=act))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_list[1]))
    model.add(tf.keras.layers.Conv1D(filters=filters_list[5], kernel_size=kernel, activation=act))
    model.add(tf.keras.layers.MaxPool1D(strides=stride))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout_list[2]))
    model.add(tf.keras.layers.GlobalAvgPool1D())
    model.add(tf.keras.layers.Dense(dropout_list, activation='sigmoid'))
    # softmax -> sigmoid 
    
    return model

def LSTM_conv1D():
    
    #########################  VARIABLES  ######################
    cell_no = 13
    Epoch_size = 50
    Lambda = 0.029  # dropout variable for regularization
    # No. of LSTM layers =2
    # Cost func. = cosh
    
    ####################   MODEL STRUCTURE  ####################
    visible=Input(shape=(None,13))
    hidden11 = LSTM(cell_no,return_sequences=True)(visible)
    hidden1=Dropout(Lambda)(hidden11)
    
    hidden22 = LSTM(cell_no)(hidden1)
    hidden2=Dropout(Lambda)(hidden22)
    
    hidden33 = Dense(10, activation='relu')(hidden2)
    hidden3 = Dropout(Lambda)(hidden33)
    # hidden4 = TimeDistributed(Dense(1))
    
    output = Dense(1, activation='sigmoid')(hidden3)
    # output=Dropout(Lambda)(output1)
    
    model = Model(inputs=visible, outputs=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    ##################   MODEL END ########################
    count = 0
    for dr in dropouts:
        total=0
        for ii in range(1):
            Lambda = dr
            print(dr)
    
            model.fit(x_train, y_train, epochs=Epoch_size, batch_size=10,verbose=1)
            predict=model.predict(x_test)
            y_pred=(predict>0.1)
            acc=model.evaluate(x_test,y_test,verbose=0)
            # print(predict)
    
            total += acc[1]
            print(acc)
        total /=1
        accuracy[count] = total
        count += 1
        
    return 

class CNN_conv2D_torch(nn.Module):
    def __init__(self, filter_list, filter_W, pool_W, stride_conv, stride_pool, conv_num, ow):
        super(CNN, self).__init__()
        
        """ """
        #filter_num = 16
        #filter_W=10
        #pool_W=3
        #stride_conv=1
        #stride_pool=1
        #conv_num=2
        #ow=100 #Wの初期化です。
        self.conv1 = nn.Conv2d(10,filter_list[0],(1,filter_W),stride=stride_conv)
        #  *conv_numは要件等
        self.conv2 = nn.Conv2d(filter_list[0],filter_list[1],(1,filter_W),stride=stride_conv)
        
        """ 
        self.conv1 = nn.Conv2d(10,16,6)
        self.conv2 = nn.Conv2d(16,320,6)
        """
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1,pool_W),stride=stride_pool)
        
        for i in range(1,conv_num):
            ow=int((ow-filter_list*conv_num)/stride_conv+1)
            ow=ow-pool_W+1
        
        #self.fc1 = nn.Linear(filter_num*conv_num * 1 * ow, 100)
        self.fc1 = nn.Linear(32* 1 * 78, filter_list[2])
        self.fc2 = nn.Linear(filter_list[2], 2)


    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x