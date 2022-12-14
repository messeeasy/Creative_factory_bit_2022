
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

from sklearn.svm import SVC
from scipy.signal import spectrogram
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

import glob
import itertools
import tensorflow as tf
import random

import torch
import torch.nn as nn
import noise_delet
from scipy import signal
import data_arrange


from scipy.io.wavfile import read, write
import soundfile as sf
# %%
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
df

# %%
print(df['path'][0])
breath=R"../dataset_heart_sound/AV/abnormal/68175_AV.wav"
#data,data_fs=dataloder.datalode(df['path'][0])
data,data_fs=data_arrange.datalode(breath)
print(data)
print(len(data))
print(data.shape)
#%%
w = wave.Wave_write("output_before.wav")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(data_fs)
w.writeframes(data)
w.close()
# %%

plt.figure(figsize=(6,6))
plt.plot(data)
plt.figure(figsize=(6,6))
p,v=FC_fucntion.fft_k(data, data_fs, 1000)
plt.plot(v,p)
# %%
data_std,data_mean,data_st=noise_delet.standard_deviation_np(data,4)

fp = 90       #通過域端周波数[Hz]
fs = 60      #阻止域端周波数[Hz]
gpass = 5       #通過域端最大損失[dB]
gstop = 40      #阻止域端最小損失[dB]
 
data_hig = noise_delet.highpass(data_std, data_fs, fp, fs, gpass, gstop)

fp = 120       #通過域端周波数[Hz]kotei
fs = 500      #阻止域端周波数[Hz]
gpass = 5     #通過域端最大損失[dB]
gstop = 20      #阻止域端最小損失[dB]kotei
 
data_low = noise_delet.lowpass(data_std, data_fs, fp, fs, gpass, gstop)

#data_fpass=FC_fucntion.FpassBand_1(data,data_fs,100,800)

#%%
plt_data=data_low
#fig,ax=plt.subplot()
plt.figure(figsize=(6,6))
plt.plot(data)
plt.plot(plt_data)####
plt.axhline(data_mean,color="r")
plt.axhline(-1*data_mean,color="r")
plt.axhline(2*data_st,color="g")
plt.axhline(-2*data_st,color="g")
#plt.xlim([13800,14000])
#plt.ylim([-10000,10000])
plt.figure(figsize=(6,6))
p,v=FC_fucntion.fft_k(plt_data, data_fs, 1000)####
plt.plot(v,p)
# %%
w = wave.Wave_write("output_data_after.wav")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(data_fs)
#w.writeframes(data_std.to('cpu').detach().numpy().copy())
#w.writeframes(data_low)
#write("output_data_low.wav", rate=data_fs, data=data_low)
sf.write("output_data_low.wav", plt_data, data_fs)###
w.close()

# %%
