import scipy.io.wavfile as wf
import numpy as np
import os
import pandas as pd
import glob
def get_path():
    if os.name=='posix':
        dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV/*/*.wav")]
    else:
        dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

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


