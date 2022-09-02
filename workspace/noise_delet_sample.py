
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
import tensorflow as tf
import random

import torch
import torch.nn as nn
import noise_delet
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
def datalode(path,divide=1,delay=0):
    Fs1, data1 = wf.read(path)
    data2=data1[delay:(len(data1)//divide+delay)]
    return data2,Fs1

# %%
print(df['path'][0])
data,fs=datalode(R"../dataset_heart_sound/AV/abnormal/68175_AV.wav")
print(data)
print(len(data))
print(data.shape)
#%%
w = wave.Wave_write("output_before.wav")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(fs)
w.writeframes(data)
w.close()
# %%

plt.figure(figsize=(6,6))
plt.plot(data)

# %%
data=noise_delet.standard_deviation(data)
#%%
plt.figure(figsize=(6,6))
plt.plot(data)

# %%
w = wave.Wave_write("output.wav")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(fs)
w.writeframes(data.to('cpu').detach().numpy().copy())
w.close()

# %%
