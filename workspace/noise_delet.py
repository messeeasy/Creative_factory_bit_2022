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

def save_heart_sound_train(data,data_fs,L,path,train_test,num):
    
    hd,name_wav=os.path.split(path)
    hd2,filename=os.path.split(hd)#filename    abnormal,normal

    basename_without_ext = os.path.splitext(os.path.basename(name_wav))[0]
    file=os.path.join(hd2+"_"+str(L)+"_filter",train_test)
    
    if not os.path.exists(file):
        os.makedirs(file)
    
    sf.write(os.path.join(file,filename+"_"+basename_without_ext+"_"+str(num)+".wav"),data, data_fs)

def save_heart_sound_test(data,data_fs,L,path,train_test):
    
    hd,name_wav=os.path.split(path)
    hd2,filename=os.path.split(hd)#filename    abnormal,normal

    basename_without_ext = os.path.splitext(os.path.basename(name_wav))[0]
    file=os.path.join(hd2+"_"+str(L)+"_filter",train_test)
    
    if not os.path.exists(file):
        os.makedirs(file)
    
    sf.write(os.path.join(file,filename+"_"+basename_without_ext+".wav"),data, data_fs)

def sava_Lsplit_heart_sound(data_L,split_num,L,data_fs,data_path,train_test):
    
    if train_test=='train':
        path_count=0
        for i in range(len(split_num)):
            data_count=0
            
            for j in range(len(split_num[i])):
                """ 
                print(data_path[path_count])
                
                print(data_L[i].shape)
                """
                #print("xxxx")
                #print(split_num[i][j])
                for k in range(split_num[i][j]-1):
                    
                    #print("data_count"+str(data_count))
                    path=data_path[path_count]
                    #print(path)
                    data_x=data_L[i][data_count]
                    #print(data_count)
                    save_heart_sound_train(data_x,data_fs,L,path,train_test,k)
                    data_count+=1
                
                path_count+=1
            #data_count+=j
    elif train_test=='test':
        path_count=0
        for data_x in data_L:
            save_heart_sound_test(data_x,data_fs,L,data_path[path_count],train_test)
            path_count+=1





    
            
                






