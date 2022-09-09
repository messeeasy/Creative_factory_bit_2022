#%%
import numpy as np
import wave
import pandas as pd
import matplotlib.pyplot as plt
import os 
import scipy.io.wavfile as wf
import FC_fucntion 
from scipy.signal import kaiserord, lfilter, firwin
from scipy.fftpack import fft

from sklearn.svm import SVC
from scipy.signal import spectrogram
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

import glob
import itertools
#import tensorflow as tf
import random

import torch
import torch.nn as nn
import k_fold
import noise_delet
import data_arrange
import train
import plot
import librosa
import librosa.display
import data_augment

#%%

k=2
L=25000
#model = ['CNN_conv1D','CNN_conv2D','CNN_conv2D_melspect']
model = ['CNN_conv2D_melspect']
#%%
#%%
df=data_arrange.get_path()

data=data_arrange.get_data(df)
#%%
y=data_arrange.get_label(df)


data_train,data_test, y_train, y_test,data_train_path,data_test_path= train_test_split(data,y,df['path'].values,train_size = 0.8, test_size=0.2)
del df,data,y
#%%
data_K_split,y_K_split=k_fold.k_fold(data_train,y_train,k)

#%%[markdown]
#ここでは長いデータが持っているため、Lごとに分割します。分割した際の余りは切り捨てています。
#また、今後どのデータを何分割したかを使用するため、そのリストをsplit_numに格納します。

#Since we have long data here, we split the data by L. The remainder of the split is truncated.
#And store the list in split_num for future use of what data and how many splits.
#%%
data_L_split,y_L_split,split_num=data_arrange.L_split_add(data_K_split,y_K_split,L)
del data_K_split,y_K_split
#%%
""" """
data_filter_after=[]
for data_x in data_L_split:
    data_filter_after.append(noise_delet.filter_processing(np.array(data_x),4000))
del data_L_split
#----------------------------------------------------
#%%
select_data,select_label=data_augment.get_select_PCG_2(data_filter_after,y_L_split,model)

#%%
# ---------- hyper pram model `--------------
filter_num_list =  [[16, 32, 64],[8,16,32],[4,8,16],[2,4,8]]
filter_size_list =  [[4, 8, 16, 32, 8],[4,8,16,32,16], [2,4,8,16,8], [8,16,32,64,16]]
dropout_para_list = [[0.5, 0.5, 0.6, 0.6, 0.7],[0.4, 0.5, 0.6, 0.7, 0.8]]
param_model = list(itertools.product(filter_num_list, filter_size_list, dropout_para_list))

for param in param_model:
    for i in range(len(model)):

        if  model[i]=='CNN_conv2D_melspect':
            in_channel = 1 # メルスペクトの値を取るだけの軸なので
            filter_num = param[0] # 参考資料の半分の半分
            filter_size = param[1]
            strides = [1,1] #　固定
            pool_strides = [1,1,1,1,1,1] #不使用
            dropout_para = param[2]

        lr = 0.01
        epoch = 50
        BATCH_SIZE=20
        df_fold=data_augment.create_df_k(k,select_data[i],select_label[i])

        for fold in range(k):
            trainloader,testloader=data_augment.model_setting_dataset(df_fold,fold,BATCH_SIZE,model[i])

            train_loader = trainloader
            val_loader = testloader # 本来はTrainの中のK個のうちのどれか
            test_loader = testloader

            device = torch.device("cuda:0")
            net = train.model_setting_cnn(model[i], in_channel, filter_num, filter_size, strides, pool_strides, dropout_para, device)
            history, net = train.training(net, lr, epoch, train_loader, val_loader, device)

            print(net)
            now = plot.evaluate_history(history)
            plot.test_result(net, test_loader, now, device)
        print("finished"+model[i])

