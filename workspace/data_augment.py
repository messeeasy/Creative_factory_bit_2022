
#%%
import data_arrange
import torch
import numpy as np
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

def change_dimensions(list1):
    list_change=[]
    for i in range(len(list1)):
     list_change.extend(list1[i])
    return np.array(list_change)

