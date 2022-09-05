
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
import train
import plot

#%%
EPOCH = 100
BATCH_SIZE=20
#WEIGHT_DECAY = 0.1
LEARNING_RATE = 0.5
#%%
print(os.name)
if os.name=='posix':
    dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV/*/*.wav")]
else:
    dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

#%%
df = pd.DataFrame.from_dict(dataset)

df.head()
#%%
# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()

#ppf1ファイルのE_VS関数からHzごとに分けられた情報を入手する。
def get_PCG(path,PCG_list):
    Fs1, data1 = wf.read(path)
    data2=data1[400:(len(data1)//5+400)]
    data2 = FC_fucntion.vec_nor(data2)
    pcgFFT1, vTfft1 = FC_fucntion.fft_k_N(data2, Fs1, 2000)

    E_PCG,C = FC_fucntion.E_VS_100(pcgFFT1, vTfft1, 'percentage')
    

    #test2.pyでどのように分けられているかをグラフで確認できます。
    kari_list=[]
    kari_list.append(vTfft1[C[0]:C[10]])
    kari_list.append(pcgFFT1[C[0]:C[10]])
    for i in range(len(C)-1):
        kari_list.append(vTfft1[C[i]:C[i+1]])
        kari_list.append(pcgFFT1[C[i]:C[i+1]])
    #kari_list.append(1)
    PCG_list.append(kari_list)

#%%
PCG=[]
for path in df['path']:
    get_PCG(path,PCG)

#%%
#いつでもいろんな値が使えるように全部dfにくっつける
#pcgc01の01はC[0]~C[1]
#ここからすべて連結させたdfをall_dfとして扱っている。
column='vT_all','pcg_all','vtc01','pcgc01','vtc12','pcgc12','vtc23','pcgc23','vtc34','pcgc34','vtc45','pcgc45','vtc56','pcgc56','vtc67','pcgc67','vtc78','pcgc78','vtc89','pcgc89','vtc910','pcgc910',
#%%
df_PCG=pd.DataFrame(PCG,columns=column)
#%%
all_df=pd.concat([df,df_PCG], axis=1)
del PCG
#%%
all_df.head()
#%%
def repeat_to_length(arr, length):
    """Repeats the numpy 1D array to given length, and makes datatype float"""
    result = np.empty((length, ), dtype = np.float32)
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result

#%%
for label in column:
    max_length = max(all_df[label].apply(len))
    if(max_length<=6452):
        max_length=100
    all_df[label] = all_df[label].apply(repeat_to_length, length=max_length)


#%%
x=[]

x1=np.stack(all_df['pcgc01'].values, axis=0)
x2=np.stack(all_df['pcgc12'].values, axis=0)
x3=np.stack(all_df['pcgc23'].values, axis=0)
x4=np.stack(all_df['pcgc34'].values, axis=0)
x5=np.stack(all_df['pcgc45'].values, axis=0)
x6=np.stack(all_df['pcgc56'].values, axis=0)
x7=np.stack(all_df['pcgc67'].values, axis=0)
x8=np.stack(all_df['pcgc78'].values, axis=0)
x9=np.stack(all_df['pcgc89'].values, axis=0)
x10=np.stack(all_df['pcgc910'].values, axis=0)
#%%
for i in range(len(all_df['pcg_all'])):
    x_list=[]
    x_list.append(x1[i])
    x_list.append(x2[i])
    x_list.append(x3[i])
    x_list.append(x4[i])
    x_list.append(x5[i])
    x_list.append(x6[i])
    x_list.append(x7[i])
    x_list.append(x8[i])
    x_list.append(x9[i])
    x_list.append(x10[i])
    x.append(x_list)
print(x)
#%%
x=np.array(x)
#%%
print(x.shape)
#%%

Y = np.stack(all_df['label'].values, axis=0)
y=np.zeros(len(Y))
for i in range(len(Y)):
    if Y[i]=='normal':
        y[i]=0
    elif Y[i]=='abnormal':
        y[i]=1

y=np.array(y)
#%%
y
#%%
"""
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32) 
"""
#x = x[:,:,np.newaxis,:]
x=x[:,:,np.newaxis,:]
x = torch.FloatTensor(x)
y = torch.LongTensor(y)

#%%
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, y, all_df['path'].values, train_size = 0.7, test_size=0.3)
#print("x_train: {0}, x_test: {1}".format(x_train.shape, x_test.shape))
del x,y

#%%
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

X_sample, y_sample = train_dataset[0]
print(X_sample.size(), y_sample.size())
#%%

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                        shuffle = True, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                        shuffle = False, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
# %%


#%% 
# ------------------------- hyper param -----------------------------
model = 'CNN_conv2D'
in_channel = 10
filter_num = [16, 16, 16, 32, 32, 32]
filter_size = [4,8,8,8,12,12]
strides = [1,1,1,1,1,1]
pool_strides = [1,1,1,1,1,1]
dropout_para = [0.2,0.2,0.2]
lr = 0.01
epoch = 100
train_loader = trainloader
val_loader = testloader # 本来はTrainの中のK個のうちのどれか
test_loader = testloader
# -------------------------------------------------------------------

device = torch.device("cuda:0")
net = train.model_setting_cnn(model, in_channel, filter_num, filter_size, strides, pool_strides, dropout_para, device)
history, net = train.training(net, lr, epoch, train_loader, val_loader, device)

print(x_train.shape)
print(x_test.shape)

#%%
now = plot.evaluate_history(history)
plot.test_result(net, test_loader, now, device)
# %%
