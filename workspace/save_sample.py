#%%[markdown]
#データの読み込み、フィルター、訓練データとテストデータの分割、訓練データのK-Fold分割、データの長さをLに分割のサンプルコード
#Sample code for reading data, filtering, splitting training data and test data, K-Fold splitting of training data, and splitting data length to L
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

import data_arrange
import noise_delet
import k_fold

#%%[markdown]
#データセットのpathを習得、OSごとに規格が違うため、それにも対応。
#Mastering the PATH of the dataset, and dealing with the different standards for each OS.
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

#%%[markdown]
#pathから信号dataを入手します。
#また、フィルターを通す時、このループ内で一緒にループさせます。
#Get signal DATA from path.
#and loop them together in this loop when passing through the filter.

#%%
""" """
data=[]
for path in df['path']:
    data_x,data_fs=data_arrange.datalode(path)
    #data_std,me,st=noise_delet.standard_deviation(data_x,2)
    
    fp_high = 90       #通過域端周波数[Hz]
    fs_high = 60      #阻止域端周波数[Hz]
    gpass_high = 5       #通過域端最大損失[dB]
    gstop_high = 40      #阻止域端最小損失[dB]
 
    #data_hig = noise_delet.highpass(data_std, data_fs, fp_high, fs_high, gpass_high, gstop_high)

    fp_low = 300       #通過域端周波数[Hz]kotei
    fs_low = 1000      #阻止域端周波数[Hz]
    gpass_low = 5     #通過域端最大損失[dB]
    gstop_low = 40      #阻止域端最小損失[dB]kotei
 
 
    #data_low = noise_delet.lowpass(data_std, data_fs, fp_low, fs_low, gpass_low, gstop_low)
    data.append(data_x)
    #noise_delet.save_heart_sound(data_x,data_fs,path)
    #print(data_low.shape)

#%%[markdown]
#読み込んだデータからTrainとTestデータ分割します。
#Train and Test data are split from the read data.
#%%
data_train,data_test,data_train_path,data_test_path= train_test_split(data,df['path'].values,train_size = 0.8, test_size=0.2)
#%%[markdown]
#data_trainからK分割して、クロスバリデーションできるように分割させます。
#このときdata_K_splitはdata_k_split[*1][*2][*3]になります。 *1:Kの分割数, *2:分割後の一つあたりのtrainの数, *3:一つのデータ
#In this case, data_K_split becomes data_k_split[*1][*2][*3]. *1:number of K splits, *2:number of trains per one after splitting, *3:one data

#%%
k=5
data_K_split=k_fold.k_fold(data_train,k)
print(data_K_split[0].shape)

#%%[markdown]
#ここでは長いデータが持っているため、Lごとに分割します。分割した際の余りは切り捨てています。
#また、今後どのデータを何分割したかを使用するため、そのリストをsplit_numに格納します。

#Since we have long data here, we split the data by L. The remainder of the split is truncated.
#And store the list in split_num for future use of what data and how many splits.
#%%
L=10000

data_L_split,split_num=data_arrange.L_split(data_K_split,L)

print(data_L_split[0].shape)
data_L_split[0][1]

#%%[markdown]
#ここでは今まで処理してきたデータセットを同じdataset_heart_soundに別ファイルとして,trainとtestファイルに保存します。
#このとき、今回使用するデータの周波数はすべて4000だったので4000で固定します。
#また、trainとtestごとに関数の実行をするようにしました。

#Here we save the datasets we have processed so far as separate files in the same dataset_heart_sound,train and test files.
#In this case, the frequencies of the data used in this project were all 4000, so we fixed them at 4000.
#I also made sure to run the function for each train and test.
#%%
#train or test
noise_delet.sava_Lsplit_heart_sound(data_L_split,split_num,L,4000,data_train_path,"train")
#%%
noise_delet.sava_Lsplit_heart_sound(data_test,split_num,L,4000,data_test_path,"test")



