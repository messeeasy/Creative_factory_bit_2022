
#%%
import data_arrange
import torch
import numpy as np
import pandas as pd
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
def select_PCG(data,model):
    if model == 'CNN_conv2D':
        data_PCG= data_arrange.get_PCG(data)
        data_PCG=np.array(data_PCG)
        return data_PCG
    else:
        return data
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
    return trainloader,testloader