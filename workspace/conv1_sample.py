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

from scipy.io import wavfile
from sklearn.svm import SVC
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

import glob
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import random
# %%
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
#%%
def datalode(path,divide=1,delay=0):
    Fs1, data1 = wf.read(path)
    data2=data1[delay:(len(data1)//divide+delay)]
    return data2,Fs1

# %%
data=[]
for i in df['path']:
    dataload=[]
    datal,fsl=datalode(i)

    dataload.append(datal)
    dataload.append(fsl)
    data.append(dataload)

del datal,fsl


#%%
#いつでもいろんな値が使えるように全部dfにくっつける
#pcgc01の01はC[0]~C[1]
#ここからすべて連結させたdfをall_dfとして扱っている。
column='x','fs'
#%%
df_x=pd.DataFrame(data,columns=column)
#%%
all_df=pd.concat([df,df_x], axis=1)
del df_x
#%%
all_df.head()
# %% [markdown]
# # Sample class audio plot 

# %% [markdown]
# Displaying all three kinds of audio classes normal, abnormal and extrasystole. We can see different patters in various classes.
# 

# %%
#Choosing one of the each samples form each catogery 
normal = all_df[all_df['label'] == 'normal' ].sample(1)
abnormal = all_df[all_df['label'] == 'abnormal' ].sample(1)
#extrasystole = df[df['label'] == 'extsys' ].sample(1)

# Plot the three samples onto three different figures
plt.figure(1, figsize=(10,5))
plt.title('normal')
plt.plot(normal['x'].values[0], c='m')

plt.figure(2, figsize=(10, 5))
plt.title('abnormal')
plt.plot(abnormal['x'].values[0], c='c')

#plt.figure(3, figsize=(10, 5))
#plt.title('extrasystole')
#plt.plot(extrasystole['x'].values[0], c='b')

# %%
#make the lenght of all audio files same by repeating audio file contents till its length is equal to max length audio file
max_length = max(all_df['x'].apply(len))

# Kaggle: What's in a heartbeat? - Peter Grenholm
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

all_df['x'] = all_df['x'].apply(repeat_to_length, length=max_length)
all_df.head()

# %%
# Collect one sample from each of the three classes and plot their waveforms
normal = all_df[all_df['label'] == 'normal' ].sample(1)
abnormal = all_df[all_df['label'] == 'abnormal' ].sample(1)
#extrasystole = df[df['label'] == 'extsys' ].sample(1)

plt.figure(1, figsize=(15,8))
plt.plot(normal['x'].values[0], c='b', label='normal', alpha=0.8)
plt.plot(abnormal['x'].values[0], c='r', label='abnormal', alpha=0.8)
#plt.plot(extrasystole['x'].values[0], c='g', label='extrasystole', alpha=0.8)

plt.title('Heartbeat waveforms overlayed onto one another')
plt.legend(loc='lower right')
# plt.savefig('temp.png')

#%%
fs = 4000
f_normal, t_normal, Sxx_normal = spectrogram(normal['x'].values[0], 4000)
plt.figure(1, figsize=(20,5))
plt.title('Normal')
plt.pcolormesh(t_normal, f_normal, Sxx_normal, cmap='Spectral')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

f_abnormal, t_abnormal, Sxx_abnormal = spectrogram(abnormal['x'].values[0], 4000)
plt.figure(2, figsize=(20, 5))
plt.title('abnormal')
plt.pcolormesh(t_abnormal, f_abnormal, Sxx_abnormal, cmap='Spectral')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

#%%
#つなげた部分pcgc01などの長さもすべて一緒にする。
#(もしかしたらいらない)

#%%
# Put the data into numpy arrays. Most machine learning libraries use numpy arrays.

""" """
x = np.stack(all_df['x'].values, axis=0)
y = np.stack(all_df['label'].values, axis=0)


""" 
#np.hstackで連結できる。
#今回は25Hz~240までをつなげて一つにしている。
x=np.hstack([np.stack(all_df['pcgc01'].values, axis=0),np.stack(all_df['pcgc12'].values, axis=0),np.stack(all_df['pcgc23'].values, axis=0),np.stack(all_df['pcgc34'].values, axis=0),np.stack(all_df['pcgc45'].values, axis=0),np.stack(all_df['pcgc56'].values, axis=0)])

y = np.stack(all_df['label'].values, axis=0)
"""
# %%
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, y, all_df['path'].values, train_size = 0.7, test_size=0.3)
#print("x_train: {0}, x_test: {1}".format(x_train.shape, x_test.shape))
print(x_train)

# %%
clf = SVC()


# %%
clf.fit(x_train, y_train)


# %%
predictions = clf.predict(x_test)
print("Accuracy %.3f" % accuracy_score(y_test, predictions))

#%%
#clf = MLPClassifier(hidden_layer_sizes=(1024,512,256,128,), max_iter=5000, verbose=True)
clf = MLPClassifier(hidden_layer_sizes=(2000,1000,500,250,), 
                    max_iter=5000, verbose=True)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print("Accuracy %.3f" % accuracy_score(y_test, predictions))
# %%
print(x_train.shape)

#%%
# Convert data to the format tf.keras expects
x_train = x_train[:,:,np.newaxis]
x_test = x_test[:,:,np.newaxis]
x_train.shape

# %% [markdown]
# # CNN Model

# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=4, kernel_size=4, activation='relu',
                input_shape = x_train.shape[1:]))
model.add(tf.keras.layers.MaxPool1D(strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPool1D(strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPool1D(strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPool1D(strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPool1D(strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.6))
#model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu'))
#model.add(tf.keras.layers.MaxPool1D(strides=2))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.7))
model.add(tf.keras.layers.GlobalAvgPool1D())
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# %%
# Need to convert y labels into one-hot encoded vectors
y_train_int_categories, y_train_class_names = pd.factorize(y_train)
print("y_train_class_names: {0}".format(y_train_class_names))

y_test_int_categories, y_test_class_names = pd.factorize(y_test)
print("y_test_class_names: {0}".format(y_test_class_names))
y_train_hot = tf.keras.utils.to_categorical(y_train_int_categories)
y_test_hot = tf.keras.utils.to_categorical(y_test_int_categories)

hist = model.fit(x_train, y_train_hot, 
                epochs=300,
                validation_data=(x_test, y_test_hot), verbose=1)

# %% [markdown]
# # Plot of accuracy and loss

# %%
#print(hist.history)
accuracy = hist.history['accuracy']
loss = hist.history['loss']
val_accuracy = hist.history['val_accuracy']
val_loss = hist.history['val_loss']

plt.figure(1, figsize=(10, 5))
plt.title('CNN - Accuracy Curves')
plt.plot(accuracy, c='m')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure(2, figsize=(10, 5))
plt.title('CNN - Loss Curves')
plt.plot(loss, c='m')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.figure(3, figsize=(10, 5))
plt.title('CNN - val_Accuracy Curves')
plt.plot(val_accuracy, c='m')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure(4, figsize=(10, 5))
plt.title('CNN - val_Loss Curves')
plt.plot(val_loss, c='m')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()