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
import data_augment
import seaborn as sns
# ------------------ noise del hyper param ---------------------
std_scale = 2
fp_l = 300       #通過域端周波数[Hz]kotei
fs_l = 1000      #阻止域端周波数[Hz]
gpass_l = 5     #通過域端最大損失[dB]
gstop_l = 40      #阻止域端最小損失[dB]kotei
#L=10000
# -------------------------------------------------------------
#%%
EPOCH = 50
BATCH_SIZE=30
#WEIGHT_DECAY = 0.1
LEARNING_RATE = 0.5
k=5
L=15000
#%%
spe=['AV','MV','PV','TV']
#%%
for sp in spe:
    df=data_arrange.get_path_V(sp)

    data=data_arrange.get_data(df)

    y=data_arrange.get_label(df)


    data_train,data_test, y_train, y_test,data_train_path,data_test_path= train_test_split(data,y,df['path'].values,train_size = 0.8, test_size=0.2)
    del df,data,y

    data_K_split,y_K_split=k_fold.k_fold(data_train,y_train,k)


    data_L_split,y_L_split,split_num=data_arrange.L_split_add(data_K_split,y_K_split,L)
    del data_K_split,y_K_split

    """ """
    data_filter_after=[]
    for data_x in data_L_split:
        data_filter_after.append(noise_delet.filter_processing(np.array(data_x),4000))

    data_augmentation2,y_augment=data_augment.get_tt_augment(data_filter_after,y_L_split)

    """ 
    noise_delet.sava_Lsplit_heart_sound(data_augmentation2,split_num,L,4000,data_train_path,"train")
    noise_delet.sava_Lsplit_heart_sound(data_test,split_num,L,4000,data_test_path,"test")
    """
    df_fold=data_augment.create_df_k(k,data_augmentation2,y_augment)

    def dataset(df_fold,fold,BATCH_SIZE=20):
        y_df = df_fold[df_fold.kfold == fold]
        x_df =  df_fold[df_fold.kfold != fold]
 
        X_valid = np.array(data_augment.change_dimensions(y_df['data'].values))
        X_train =  np.array(data_augment.change_dimensions(x_df['data'].values))

        y_valid = np.array(data_augment.change_dimensions(y_df['label_y'].values))
        y_train = np.array(data_augment.change_dimensions(x_df['label_y'].values))
    
    
        return X_train,X_valid,y_train,y_valid


    for fold in range(k):
        X_train,X_valid,y_train,y_valid=dataset(df_fold,fold,BATCH_SIZE=20)
        #print(len(X_train[0]))
        """ """
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)
        print("Accuracy %.3f" % accuracy_score(y_valid, predictions))
        cm = confusion_matrix(predictions,y_valid)
        sns.heatmap(cm,annot=True, cbar=False, square=True, fmt="d")
        print("finished")

    for fold in range(k):
        X_train,X_valid,y_train,y_valid=dataset(df_fold,fold,BATCH_SIZE=20)
        clf = MLPClassifier(hidden_layer_sizes=(2000,1000,500,250,), 
                    max_iter=5000, verbose=True)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)
        print("Accuracy %.3f" % accuracy_score(y_valid, predictions))
        cm = confusion_matrix(predictions,y_valid)
        sns.heatmap(cm,annot=True, cbar=False, square=True, fmt="d")
        print("finished")

    for fold in range(k):
        in_channel = 1
        filter_num = [4, 8, 16, 32, 64, 128]
        filter_size = [4,4,4,4,4,14]
        strides = [1,1,1,1,1,1]
        pool_strides = [2,2,2,2,2,2]
        dropout_para = [0.2,0.2,0.2,0.2,0.2,0.2]
        lr = 0.01
        epoch = 100
        trainloader,testloader=data_augment.model_setting_dataset(df_fold,fold,BATCH_SIZE,'CNN_conv1D')

        train_loader = trainloader
        val_loader = testloader # 本来はTrainの中のK個のうちのどれか
        test_loader = testloader

        device = torch.device("cuda:0")
        net = train.model_setting_cnn('CNN_conv1D', in_channel, filter_num, filter_size, strides, pool_strides, dropout_para, device)
        history, net = train.training(net, lr, epoch, train_loader, val_loader, device)

        print(net)
        now = plot.evaluate_history(history)
        plot.test_result(net, test_loader, now, device)
        print("finished")
