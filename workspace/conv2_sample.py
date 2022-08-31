
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
#%%
EPOCH = 20
BATCH_SIZE=40
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.1
#%%
print(os.name)
if os.name=='posix':
    dataset = [{'path': path, 'label': path.split('/' )[3] } for path in glob.glob("../dataset_heart_sound/AV/*/*.wav")]
else:
    dataset = [{'path': path, 'label': path.split('\\' )[3] } for path in glob.glob("..\dataset_heart_sound\AV\**\*.wav")]

#%%
df = pd.DataFrame.from_dict(dataset)

df.head()
#%%
# Add a column to store the data read from each wavfile...   
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()

#ppf1ファイルのE_VS関数からHzごとに分けられた情報を入手する。
def get_PCG(path,PCG_list):
    Fs1, data1 = wf.read(path)

    data1 = FC_fucntion.vec_nor(data1)
    pcgFFT1, vTfft1 = FC_fucntion.fft_k_N(data1, Fs1, 2000)

    E_PCG,C = FC_fucntion.E_VS_100(pcgFFT1, vTfft1, 'percentage')
    

    #test2.pyでどのように分けられているかをグラフで確認できます。
    kari_list=[]
    kari_list.append(vTfft1[C[0]:C[10]])
    kari_list.append(pcgFFT1[C[0]:C[10]])
    for i in range(len(C)-1):
        kari_list.append(vTfft1[C[i]:C[i+1]])
        kari_list.append(pcgFFT1[C[i]:C[i+1]])
    #kari_list.append(1)
    PCG_list.append(kari_list)

#%%
PCG=[]
for path in df['path']:
    get_PCG(path,PCG)

#%%
#いつでもいろんな値が使えるように全部dfにくっつける
#pcgc01の01はC[0]~C[1]
#ここからすべて連結させたdfをall_dfとして扱っている。
column='vT_all','pcg_all','vtc01','pcgc01','vtc12','pcgc12','vtc23','pcgc23','vtc34','pcgc34','vtc45','pcgc45','vtc56','pcgc56','vtc67','pcgc67','vtc78','pcgc78','vtc89','pcgc89','vtc910','pcgc910',
#%%
df_PCG=pd.DataFrame(PCG,columns=column)
#%%
all_df=pd.concat([df,df_PCG], axis=1)
del PCG
#%%
all_df.head()
#%%
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

#%%
for label in column:
    max_length = max(all_df[label].apply(len))
    if(max_length<=6452):
        max_length=100
    all_df[label] = all_df[label].apply(repeat_to_length, length=max_length)


#%%
x=[]

x1=np.stack(all_df['pcgc01'].values, axis=0)
x2=np.stack(all_df['pcgc12'].values, axis=0)
x3=np.stack(all_df['pcgc23'].values, axis=0)
x4=np.stack(all_df['pcgc34'].values, axis=0)
x5=np.stack(all_df['pcgc45'].values, axis=0)
x6=np.stack(all_df['pcgc56'].values, axis=0)
x7=np.stack(all_df['pcgc67'].values, axis=0)
x8=np.stack(all_df['pcgc78'].values, axis=0)
x9=np.stack(all_df['pcgc89'].values, axis=0)
x10=np.stack(all_df['pcgc910'].values, axis=0)
#%%
for i in range(len(all_df['pcg_all'])):
    x_list=[]
    x_list.append(x1[i])
    x_list.append(x2[i])
    x_list.append(x3[i])
    x_list.append(x4[i])
    x_list.append(x5[i])
    x_list.append(x6[i])
    x_list.append(x7[i])
    x_list.append(x8[i])
    x_list.append(x9[i])
    x_list.append(x10[i])
    x.append(x_list)
print(x)
#%%
x=np.array(x)
#%%
print(x.shape)
#%%

Y = np.stack(all_df['label'].values, axis=0)
y=np.zeros(len(Y))
for i in range(len(Y)):
    if Y[i]=='normal':
        y[i]=0
    elif Y[i]=='abnormal':
        y[i]=1

y=np.array(y)
#%%
y
#%%
"""
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32) 
"""
#x = x[:,:,np.newaxis,:]
x=x[:,:,np.newaxis,:]
x = torch.FloatTensor(x)
y = torch.LongTensor(y)
#%%
#%%
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, y, all_df['path'].values, train_size = 0.7, test_size=0.3)
#print("x_train: {0}, x_test: {1}".format(x_train.shape, x_test.shape))

#%%
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

X_sample, y_sample = train_dataset[0]
print(X_sample.size(), y_sample.size())
#%%

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE,
                        shuffle = True, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE,
                        shuffle = False, num_workers = 0) #Windows Osの方はnum_workers=1 または 0が良いかも
# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        """ """
        fillter_num = 16
        fillter_W=20
        pool_W=3
        stride_conv=1
        stride_pool=1
        conv_num=2
        ow=100#Wの初期化です。
        self.conv1 = nn.Conv2d(10,fillter_num,(1,fillter_W),stride=stride_conv)
        #  *conv_numは要件等
        self.conv2 = nn.Conv2d(fillter_num,fillter_num*conv_num,(1,fillter_W),stride=stride_conv)
        
        """ 
        self.conv1 = nn.Conv2d(10,16,6)
        self.conv2 = nn.Conv2d(16,320,6)
        """
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1,pool_W),stride=stride_pool)
        
        for i in range(1,conv_num):
            ow=int((ow-fillter_num*conv_num)/stride_conv+1)
            ow=ow-pool_W+1
        #self.fc1 = nn.Linear(fillter_num*conv_num * 1 * ow, 100)
        self.fc1 = nn.Linear(32* 1 * 58, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = x.view(x.size()[0], -1)
        print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
#%%
import torch.optim as optim
device = torch.device("cuda:0")
net = CNN()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


#%%
print(net)
# %%
print(criterion)
# %%
print(optimizer)

#%%
train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist

# %%
for epoch in range(EPOCH):
    print('epoch', epoch+1)    #epoch数の出力
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計

    #train dataを使ってテストをする(パラメータ更新がないようになっている)
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
    print("train mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(trainloader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    #test dataを使ってテストをする
    for (inputs, labels) in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    print("test  mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(testloader.dataset), float(sum_correct/sum_total)))
    test_loss_value.append(sum_loss*BATCH_SIZE/len(testloader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))

plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")
# %%
