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
import data_arrange
import noise_delet
import torch
import k_fold
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
# %%
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()

#%%

""" """
data=[]
for path in df['path']:
    data_x,data_fs=data_arrange.datalode(path)
    #data_std,me,st=noise_delet.standard_deviation(data_x,2)
    
    fp = 90       #通過域端周波数[Hz]
    fs = 60      #阻止域端周波数[Hz]
    gpass = 5       #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]

    #data_hig = noise_delet.highpass(data_std, data_fs, fp, fs, gpass, gstop)

    fp = 300       #通過域端周波数[Hz]kotei
    fs = 1000      #阻止域端周波数[Hz]
    gpass = 5     #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]kotei


    #data_low = noise_delet.lowpass(data_std, data_fs, fp, fs, gpass, gstop)
    data.append(data_x)
    #noise_delet.save_heart_sound(data_x,data_fs,path)
    #print(data_low.shape)

#%%
data_train,data_test= train_test_split(data,train_size = 0.8, test_size=0.2)
k=5
data_K_split=k_fold.k_fold(data_train,k)
#%%
print(data_K_split[0].shape)

#%%
L=10000

data_L_split,split_num=data_arrange.L_split(data_K_split,L)


# %%
print(data_L_split[4].shape)

# %%
