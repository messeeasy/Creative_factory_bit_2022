
#%%
import data_arrange
import torch
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
#%%
def cnn_conv1_dataset(df_fold,fold,BATCH_SIZE=20):
    y_df = df_fold[df_fold.kfold == fold]
    x_df =  df_fold[df_fold.kfold != fold]
 
    X_valid = np.array(change_dimensions(y_df['data'].values))
    X_train =  np.array(change_dimensions(x_df['data'].values))

    y_valid = np.array(change_dimensions(y_df['label_y'].values))
    y_train = np.array(change_dimensions(x_df['label_y'].values))
    X_train=X_train[:,np.newaxis,:]
    X_valid=X_valid[:,np.newaxis,:]
    #print(type(X_train))
    #print(X_train.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.astype('float64')), torch.LongTensor(y_train.astype('int16')))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid.astype('float64')), torch.LongTensor(y_valid.astype('float16')))
    X_sample, y_sample = train_dataset[0]
    print(X_sample.size(), y_sample.size())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                        shuffle = True, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                        shuffle = False, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    
    return trainloader,testloader
def cnn_conv2_dataset(df_fold,fold,BATCH_SIZE=20):
    y_df = df_fold[df_fold.kfold == fold]
    x_df =  df_fold[df_fold.kfold != fold]
 
    X_valid = np.array(change_dimensions(y_df['data'].values))
    X_train =  np.array(change_dimensions(x_df['data'].values))

    y_valid = np.array(change_dimensions(y_df['label_y'].values))
    y_train = np.array(change_dimensions(x_df['label_y'].values))
    X_train=X_train[:,:,np.newaxis,:]
    X_valid=X_valid[:,:,np.newaxis,:]
    #print(type(X_train))
    #print(X_train.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.astype('float64')), torch.LongTensor(y_train.astype('int16')))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid.astype('float64')), torch.LongTensor(y_valid.astype('float16')))
    X_sample, y_sample = train_dataset[0]
    print(X_sample.size(), y_sample.size())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                        shuffle = True, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                        shuffle = False, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    
    return trainloader,testloader
def cnn_conv2_melsp_dataset(df_fold,fold,BATCH_SIZE=20):
    y_df = df_fold[df_fold.kfold == fold]
    x_df =  df_fold[df_fold.kfold != fold]
 
    X_valid = np.array(change_dimensions(y_df['data'].values))
    X_train =  np.array(change_dimensions(x_df['data'].values))

    y_valid = np.array(change_dimensions(y_df['label_y'].values))
    y_train = np.array(change_dimensions(x_df['label_y'].values))
    X_train=X_train[:,np.newaxis,:,:]
    X_valid=X_valid[:,np.newaxis,:,:]
    #print(type(X_train))
    #print(X_train.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.astype('float64')), torch.LongTensor(y_train.astype('int16')))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_valid.astype('float64')), torch.LongTensor(y_valid.astype('float16')))
    X_sample, y_sample = train_dataset[0]
    print(X_sample.size(), y_sample.size())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                        shuffle = True, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                        shuffle = False, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
    
    return trainloader,testloader
def change_dimensions(list1):
    list_change=[]
    for i in range(len(list1)):
     list_change.extend(list1[i])
    return np.array(list_change)
def create_df_k(k,data_filter_after,y_L_split):
    pd_fold=np.array(list(range(k)))
    
    df_fold = pd.DataFrame(pd_fold,
                  columns=['kfold'],)
    #df_fold['data']=data_L_split
    df_fold['data']=data_filter_after
    df_fold['label_y']=y_L_split
    return df_fold
def get_melsp(data_filter_after):
    melpse_data=[]
    for data in data_filter_after:
        data2=[]
        for data_x in data:
            data2.append(calculate_melsp(data_x))
        melpse_data.append(data2)
    melpse_data=np.array(melpse_data)
    return melpse_data
def select_PCG(data,model):
    if model == 'CNN_conv2D':
        data_PCG= data_arrange.get_PCG(data)
        data_PCG=np.array(data_PCG)
        return data_PCG
    if model =='CNN_conv2D_melsp':
        data_melsp=get_melsp(data)
        return data_melsp
    else:
        return data
def calculate_melsp(x, n_fft=1024, hop_length=128):
    #print(len(x))
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    #show_melsp(melsp, 4000)
    #print(melsp)
    return melsp

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, x_axis="time", y_axis="mel", hop_length=128)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()
def get_melsp(data):
    melpse=[]
    for data_x in data:
        melpse.append(calculate_melsp(data_x, n_fft=1024, hop_length=128))
    return melpse
def get_select_PCG(data_filter_after,model):
    select_data=[]
    for i in range(len(model)):
        select_data.append(select_PCG(data_filter_after,model[i]))
    return select_data
def model_setting_dataset(df_fold,fold,BATCH_SIZE,model):
    if model == 'CNN_conv1D':
        trainloader,testloader=cnn_conv1_dataset(df_fold,fold,BATCH_SIZE)
    elif model == 'CNN_conv2D':
        trainloader,testloader=cnn_conv2_dataset(df_fold,fold,BATCH_SIZE)
    elif model == 'CNN_conv2D_melsp':
        trainloader,testloader=cnn_conv2_melsp_dataset(df_fold,fold,BATCH_SIZE)
    return trainloader,testloader


def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")