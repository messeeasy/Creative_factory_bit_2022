import torch
from scipy import signal
import numpy as np
import soundfile as sf
import os
def standard_deviation(data,N=1):
    data = torch.FloatTensor(data)
    data_mean=torch.mean(abs(data))
    data_std=torch.std(abs(data))
    #print(data_std)
    #print(data_mean)

    for i in range(len(data)):

        if abs(data[i])>N*data_std:
            if data[i]>0:
                data[i]=data_mean
            else:
                data[i]=-1*data_mean
        
    return np.array(data),data_mean,data_std

def highpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "high")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す
def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す

def save_heart_sound(data,data_fs,path):
    
    hd,name_wav=os.path.split(path)
    hd2,filename=os.path.split(hd)
    hd3,file_V_name=os.path.split(hd)
    
    file=os.path.join(hd3+"_filter",file_V_name)
    
    if not os.path.exists(file):
        os.makedirs(file)
    
    if filename=='normal':

        sf.write(os.path.join(file,name_wav),data, data_fs)
    else:

        sf.write(os.path.join(file,name_wav),data, data_fs)



