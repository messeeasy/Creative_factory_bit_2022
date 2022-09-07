#%%[markdown]
##import library
#%%

from statistics import mean
import numpy as np
import wave
import pandas as pd
import matplotlib.pyplot as plt
import os 
import scipy.io.wavfile as wf
import FC_fucntion
from scipy.signal import kaiserord, lfilter, firwin
from scipy.fftpack import fft

from scipy.io import wavfile
from sklearn.svm import SVC
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

import glob
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torch.optim as optim

import data_arrange
import noise_delet
import k_fold
import models
import train
import data_augment
import plot
import data_augment
#%%
df=data_arrange.get_path()

data=data_arrange.get_data(df)
#%%
Y = np.stack(df['label'].values, axis=0)
y=np.zeros(len(Y))
for i in range(len(Y)):
    if Y[i]=='normal':
        y[i]=0
    elif Y[i]=='abnormal':
        y[i]=1

y=np.array(y)
data_train,data_test, y_train, y_test,data_train_path,data_test_path= train_test_split(data,y,df['path'].values,train_size = 0.8, test_size=0.2)
#%%
k=5
data_K_split,y_K_split=k_fold.k_fold(data_train,y_train,k)
print(data_K_split[0].shape)

#%%[markdown]
#ここでは長いデータが持っているため、Lごとに分割します。分割した際の余りは切り捨てています。
#また、今後どのデータを何分割したかを使用するため、そのリストをsplit_numに格納します。

#Since we have long data here, we split the data by L. The remainder of the split is truncated.
#And store the list in split_num for future use of what data and how many splits.
#%%
L=10000

data_L_split,y_L_split,split_num=data_arrange.L_split_add(data_K_split,y_K_split,L)

print(data_L_split[0].shape)
print(len(y_L_split[0]))
data_L_split[0][1]
#%%
""" """
data_filter_after=[]
for data_x in data_L_split:
    
    print("0")
    data_filter_after.append(noise_delet.filter_processing(np.array(data_x),4000))
#data_filter_after=np.array(data_filter_after)
del data_L_split

#%%

pd_fold=np.array(list(range(k)))
df_fold = pd.DataFrame(pd_fold,
                  columns=['kfold'],)
#df_fold['data']=data_L_split
df_fold['data']=data_filter_after
df_fold['label_y']=y_L_split
df_fold.head()

#%%
model = 'CNN_conv1D'
in_channel = 1
filter_num = [4, 8, 16, 32, 64, 128]
filter_size = [4,4,4,4,4,14]
strides = [1,1,1,1,1,1]
pool_strides = [2,2,2,2,2,2]
dropout_para = [0.2,0.2,0.2]
lr = 0.01
epoch = 100
BATCH_SIZE=20
#%%
for fold in range(k):
    trainloader,testloader=data_augment.cnn_conv1_dataset(df_fold,fold,BATCH_SIZE)

    train_loader = trainloader
    val_loader = testloader # 本来はTrainの中のK個のうちのどれか
    test_loader = testloader

    device = torch.device("cuda:0")
    net = train.model_setting_cnn(model, in_channel, filter_num, filter_size, strides, pool_strides, dropout_para, device)
    history, net = train.training(net, lr, epoch, train_loader, val_loader, device)

    print(net)
    now = plot.evaluate_history(history)
    plot.test_result(net, test_loader, now, device)
# %%


