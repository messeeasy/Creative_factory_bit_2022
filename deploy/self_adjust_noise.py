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
import noise_delete
import data_arrange
import train
import plot
import librosa
import librosa.display
import random


#%%
# 
if os.name=='posix':
    dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV_param/*/*.wav")]
else:
    dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV_param\**\*.wav")]

#%%
df = pd.DataFrame.from_dict(dataset)

df.head()
#%%
# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()
normal_index = [ni for ni, n in enumerate(df['label']) if n == 'normal']
abnormal_index = [abi for abi, ab in enumerate(df['label']) if ab == 'abnormal']

# %%
random_normal_index = random.sample(normal_index, k = 5)
random_abnormal_index =random.sample(abnormal_index, k =5)
print(random_normal_index)

# %%
# ----------------- メルスペクト前処理関数 --------------------
def datalode(path,length,delay=0):
    #Fs1, data1 = wf.read(path)
    data1, Fs1 = librosa.load(path)
    data2=data1[delay:(length+delay)]
    return np.array(data2),Fs1

# メルスペクト
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, cmap='magma',x_axis="time", y_axis="mel", hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()
# -----------------------------------------------------------

#%%
# ------------------ noise del hyper param ---------------------
length = [150000]
delay = [0]
std_scale = [4] #信号の標準偏差のstd_scale倍以上の値を平均値に変換する
fp_l = [120] #通過域端周波数[Hz]
fs_l = [500] #阻止域端周波数[Hz]
gpass_l = [5] #通過域端最大損失[dB]
gstop_l = [20]  #阻止域端最小損失[dB]
# ---------------------------------------------------------------
param_noise = list(itertools.product(length, delay, std_scale, fp_l, fs_l, gpass_l, gstop_l))
param_noise = [p for p in param_noise if p[3] < p[4]]

param = param_noise[0]
data,data_fs=datalode(df['path'][1],param[0],param[1])
melsp = calculate_melsp(data)

dataset_nothing = []
dataset_melsp = []
dataset_del_noise_signal = []
dataset_index = []
dataset_fs = []
dataset_index.extend(random_normal_index)
dataset_index.extend(random_abnormal_index)


for i in dataset_index:
    path = df['path'][i]
    # 長さを調節するため、いったんコメントアウト
    #data,data_fs=data_arrange.datalode(path)
    data,data_fs=datalode(path,param[0],param[1])
    data_del,me,st=noise_delete.standard_deviation_np(data, param[2])
    data_del = noise_delete.lowpass(data_del, data_fs, param[3], param[4], param[5], param[6])
    # 周波数変換コード　使わないとき除く
    #data = get_PCG_noise_del(data, data_fs)
    melsp = calculate_melsp(data_del)
    dataset_nothing.append(data)
    dataset_melsp.append(melsp)
    dataset_del_noise_signal.append(data_del)
    dataset_fs.append(data_fs)

for i in range(len(dataset_melsp)):
    #　メルスペクト表示
    show_melsp(dataset_melsp[i], dataset_fs[i])
    
    # 信号表示

for i in range(len(dataset_del_noise_signal)):
    if i < 5:
        plt.figure(i*2, figsize=(10,5))
        plt.title('before normal')
        plt.plot(dataset_nothing[i], c='c')
        plt.figure(i*2+1, figsize=(10,5))
        plt.title('after_processing normal')
        plt.plot(dataset_del_noise_signal[i], c='m')
        print(df['path'][dataset_index[i]])
        librosa.audio.sf.write("output_wav/del_noise_normal_" + str(i) + '.wav', dataset_del_noise_signal[i], dataset_fs[i])

    else:
        plt.figure(i*2, figsize=(10,5))
        plt.title('before abnormal')
        plt.plot(dataset_nothing[i], c='c')
        plt.figure(i*2+1, figsize=(10,5))
        plt.title('after_processing abnormal')
        plt.plot(dataset_del_noise_signal[i], c='m')
        print(df['path'][dataset_index[i]])
        librosa.audio.sf.write("output_wav/del_abnoise_normal_" + str(i) + '.wav', dataset_del_noise_signal[i], dataset_fs[i])

# %%
