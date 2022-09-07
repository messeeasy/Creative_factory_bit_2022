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
#%%
print(os.name)
if os.name=='posix':
    dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV/*/*.wav")]
else:
    dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

df = pd.DataFrame.from_dict(dataset)
df.head()
# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()
#%%
""" """




data=[]
for path in df['path']:
    data_x,data_fs=data_arrange.datalode(path) 
    data.append(data_x)

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
for data_x in data_L_split:
    data_filter_after=[]
    print("0")
    data_filter_after.append(noise_delet.filter_processing(np.array(data_x),4000))
#data_filter_after=np.array(data_filter_after)
del data_L_split

#%%

pd_fold=np.array(list(range(k)))
df_fold = pd.DataFrame(pd_fold,
                  columns=['kfold'],)
df_fold['data']=data_L_split
df_fold['label_y']=y_L_split
df_fold.head()


