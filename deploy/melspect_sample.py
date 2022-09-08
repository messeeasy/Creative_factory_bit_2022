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


#%%

BATCH_SIZE=20

#%%
#print(os.name)
if os.name=='posix':
    dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV_param/*/*.wav")]
else:
    dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

df = pd.DataFrame.from_dict(dataset)
df.head()
#%%
# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()

#%%
# ------------------ noise del hyper param ---------------------
std_scale = 2
fp_l = 300       #通過域端周波数[Hz]kotei
fs_l = 1000      #阻止域端周波数[Hz]
gpass_l = 5     #通過域端最大損失[dB]
gstop_l = 40      #阻止域端最小損失[dB]kotei
#L=10000

length = [30000]
delay = [0]
std_scale = [4]
fp_l = [120]
fs_l = [500]
gpass_l = [5]
gstop_l = [20]
n_fft = 1024
hop_length = 240
param_noise = list(itertools.product(length, delay, std_scale, fp_l, fs_l, gpass_l, gstop_l))
param_noise = [p for p in param_noise if p[3] < p[4]]
# -------------------------------------------------------------

#%%

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
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=hop_length)
    return melsp

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, cmap='magma', x_axis="time", y_axis="mel", hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()
# -----------------------------------------------------------

# example data
param = param_noise[0]
data,data_fs=datalode(df['path'][1],param[0],param[1])
melsp = calculate_melsp(data)
#show_melsp(melsp,data_fs)
    
# %%
# ------------------------ データ前処理 --------------------------
dataset_all = []

for path in df['path']:
        # 長さを調節するため、いったんコメントアウト
        #data,data_fs=data_arrange.datalode(path)
        data,data_fs=datalode(path,param[0],param[1])
        data,me,st=noise_delet.standard_deviation_np(data, param[2])
        data = noise_delet.lowpass(data, data_fs, param[3], param[4], param[5], param[6])
        # 周波数変換コード　使わないとき除く
        #data = get_PCG_noise_del(data, data_fs)
        melsp = calculate_melsp(data, n_fft, hop_length)
        dataset_all.append(melsp)
        
#%%
dataset_all = np.array(dataset_all)
print(dataset_all.shape)
show_melsp(dataset_all[1],4000)
# dataset(N, C=128, W =1, H =118)　Hは音の長さによって変わる。今回は15000でそろえている

# %%
# -------------------- pytorch make dataset -------------------------
# all_dfの意味とその代替
Y = np.stack(df['label'].values, axis=0)
y=np.zeros(len(Y))
for i in range(len(Y)):
    if Y[i]=='normal':
        y[i]=0
    elif Y[i]=='abnormal':
        y[i]=1
        
y=np.array(y)
#%%
len(dataset_all[0][0])
#%%
dataset_all = np.array(dataset_all)
x=dataset_all[:,np.newaxis,:,:]

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

# %%
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, y, df['path'].values, train_size = 0.7, test_size=0.3)
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
# ------------------------- hyper param -----------------------------
model = 'CNN_conv2D_melspect'
in_channel = 1 # メルスペクトの値を取るだけの軸なので
filter_num = [8, 16, 32] # 参考資料の半分の半分
filter_size = [4, 8, 16, 32, 8]
strides = [1,1] #　固定
pool_strides = [1,1,1,1,1,1] #不使用
dropout_para = [0.3, 0.4, 0.5, 0.6, 0.7]
lr = 0.0001
epoch = 150
train_loader = trainloader
val_loader = testloader # 本来はTrainの中のK個のうちのどれか
test_loader = testloader
# -------------------------------------------------------------------

device = torch.device("cuda:0")
net = train.model_setting_cnn(model, in_channel, filter_num, filter_size, strides, pool_strides, dropout_para, device)
history, net = train.training(net, lr, epoch, train_loader, val_loader, device)

print(x_train.shape)
print(x_test.shape)
print(net)

#%%
now = plot.evaluate_history(history)
plot.test_result(net, test_loader, now, device)
