import scipy.io.wavfile as wf
import numpy as np
import os
import pandas as pd
import glob
import FC_fucntion
def get_path():
    
    if os.name=='posix':
        dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV/*/*.wav")]
    else:
        dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

    df = pd.DataFrame.from_dict(dataset)
    return df
def get_path_V(spe):
    
    
    if os.name=='posix':
        file="../dataset_heart_sound/"
        file=os.path.join(file,spe+"/*/*.wav")
        dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob(file)]
    else:
        file="..\dataset_heart_sound"
        file=os.path.join(file,spe+"\**\*.wav")
        dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob(file)]

    df = pd.DataFrame.from_dict(dataset)
    return df
def get_df_x(df):
    df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
    #df.head()
    return  df
def datalode(path,divide=1,delay=0):
    Fs1, data1 = wf.read(path)
    data2=data1[delay:(len(data1)//divide+delay)]
    return np.array(data2),Fs1
def get_data(df):
    data=[]
    for path in df['path']:
        data_x,data_fs=datalode(path) 
        data.append(data_x)
    return data
def get_label(df):
    Y = np.stack(df['label'].values, axis=0)
    y=np.zeros(len(Y))
    for i in range(len(Y)):
        if Y[i]=='normal':
            y[i]=0
        elif Y[i]=='abnormal':
            y[i]=1

    return np.array(y)
def repeat_to_length(arr, length):
    """Repeats the numpy 1D array to given length, and makes datatype float"""
    result = np.empty((length, ), dtype = np.float32)
    l = len(arr)
    pos = 0
    while pos + l <= length:
        result[pos:pos+l] = arr
        pos += l
    if pos < length:
        result[pos:length] = arr[:length-pos]
    return result

def L_split_process(data,L=10000):
    data_L_split=[]
    split_num=[]

    for data_x in data:
        
        num=len(data_x)//L
        split_num.append(num)
        #print(len(data_x),num)
        for i in range(num-1):
            data_L_split.append(data_x[i*L:(i+1)*L])
    return  data_L_split,split_num
def L_split(data,L=10000):
    data_L_split=[]
    split_num=[]
    for data_s in data:

        data_L,split_n=L_split_process(data_s,L)
        data_L_split.append(np.array(data_L))
        split_num.append(split_n)
    return data_L_split,split_num

def L_split_add(data,y,L=10000):
    data_L_split=[]
    split_num=[]
    for data_s in data:

        data_L,split_n=L_split_process(data_s,L)
        data_L_split.append(np.array(data_L))
        split_num.append(np.array(split_n))
    y_label=[]
    for i in range(len(y)):
        y_t=[]
        for j in range(len(y[i])):
           
            
            label=[]
            
            for k in range(split_num[i][j]-1):
                y_t.append(y[i][j])
            """     ss
            label.append(y[i][j])
            y_t.append(label)
            """
                
        y_label.append(np.array(y_t))
    return np.array(data_L_split),np.array(y_label),split_num


def cal_PCG(data,Fs1=4000):
    
    data_=[]
    for data2 in data:
       
        data2 = FC_fucntion.vec_nor(np.array(data2))
        pcgFFT1, vTfft1 = FC_fucntion.fft_k_N(data2, Fs1, 1000)

        E_PCG,C = FC_fucntion.E_VS_100(pcgFFT1, vTfft1, 'percentage')
        #pcgFFT1=np.array(pcgFFT1)
        #test2.pyでどのように分けられているかをグラフで確認できます。
        kari_list=[]
        for i in range(len(C)-1):
            #np.append(kari_list,vTfft1[C[i]:C[i+1]])
            #print(pcgFFT1[C[i]:C[i+1]])
            
            #np.append(kari_list,pcgFFT1[C[i]:C[i+1]])
            kari_list.append(pcgFFT1[C[i]:C[i+1]])
        #print(kari_list)
        #np.append(data,kari_list,axis=0)
        df = pd.DataFrame(np.array(kari_list).T,
                  columns=['filter'],)
        max_length = max(df['filter'].apply(len))
        #print(max_length)
        df['filter'] = df['filter'].apply(repeat_to_length, length=max_length)
        data_.append(np.stack(df['filter'].values, axis=0))
    return data_

def get_PCG(data_L_split):
    
    data_PCG=[]
    for data in data_L_split:
        data_PCG.append(cal_PCG(data,4000))
    return data_PCG