import torch
from scipy import signal
import numpy as np
def standard_deviation(data):
    data = torch.FloatTensor(data)
    data_mean=torch.mean(data)
    data_std=torch.std(data)
    print(data_std)
    print(data_mean)

    for i in range(len(data)):
        """ 
        if abs(data[i])<abs(data_std):
            #data[i]/=data_std
            data[i]=0
        """
        """ 
        if 2*abs(data_mean)<abs(data[i]) and abs(data[i])<3*abs(data_mean):
            data[i]=data_mean
        if 2**abs(data_mean)<abs(data[i]) and abs(data[i])<3**abs(data_mean):
            data[i]=data_mean
        """
        
        """ 
        if 2*abs(data_std)<abs(data[i]) and abs(data[i])<3*abs(data_std):
            data[i]=data_mean
        if 2**abs(data_std)<abs(data[i]) and abs(data[i])<3**abs(data_std):
            data[i]=data_mean
        """
        """
        if 2*abs(data_std)*0.9<abs(data[i]) and abs(data[i])<2*abs(data_std)*1.1:
            data[i]=0
        if 3*abs(data_std)*0.9<abs(data[i]) and abs(data[i])<3*abs(data_std)*1.1:
            data[i]=0
        if 2**abs(data_std)*0.9<abs(data[i]) and abs(data[i])<2**abs(data_std)*1.1:
            data[i]=0
        if 3**abs(data_std)*0.9<abs(data[i]) and abs(data[i])<3**abs(data_std)*1.1:
            data[i]=0
        """
    return data

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