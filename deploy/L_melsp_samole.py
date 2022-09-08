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
# ------------------ noise del hyper param ---------------------
std_scale = 2
fp_l = 300       #通過域端周波数[Hz]kotei
fs_l = 1000      #阻止域端周波数[Hz]
gpass_l = 5     #通過域端最大損失[dB]
gstop_l = 40      #阻止域端最小損失[dB]kotei
#L=10000

length = [15000, 200, 250, 300, 350]
delay = [0]
std_scale = [2,2.5,3,3.5,4,4.5,5,5.5,6,7,8,9,10]
fp_l = [300, 200, 300, 400, 500, 600, 700, 800, 900]
fs_l = [1000, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
gpass_l = [3, 5, 7]
gstop_l = [20, 30, 40, 50]
param_noise = list(itertools.product(length, delay, std_scale, fp_l, fs_l, gpass_l, gstop_l))
param_noise = [p for p in param_noise if p[3] < p[4]]
# -------------------------------------------------------------
#%%
EPOCH = 50
BATCH_SIZE=30
#WEIGHT_DECAY = 0.1
LEARNING_RATE = 0.5
#%%
#%%
df=data_arrange.get_path()

data=data_arrange.get_data(df)
#%%
y=data_arrange.get_label(df)

#----------------------------------------------------

#%%
# メルスペクト
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, x_axis="time", y_axis="mel", hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()