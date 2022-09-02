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
import dataloder
import noise_delet
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
# %%
df['x'] = df['path'].apply(lambda x: wf.read(x)[1])
df.head()


# %%
#make the lenght of all audio files same by repeating audio file contents till its length is equal to max length audio file
max_length = max(df['x'].apply(len))

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

df['x'] = df['x'].apply(repeat_to_length, length=max_length)
df.head()

#%%
#つなげた部分pcgc01などの長さもすべて一緒にする。
#(もしかしたらいらない)

#%%

""" """
data=[]
for path in df['path']:
    data_x,data_fs=dataloder.datalode(path)
    
    fp = 100       #通過域端周波数[Hz]
    fs = 70      #阻止域端周波数[Hz]
    gpass = 5       #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]
 
    data_hig = noise_delet.highpass(data_x, data_fs, fp, fs, gpass, gstop)

    fp = 300       #通過域端周波数[Hz]kotei
    fs = 400      #阻止域端周波数[Hz]
    gpass = 5     #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]kotei
 
    data_low = noise_delet.lowpass(data_hig, data_fs, fp, fs, gpass, gstop)
    data.append(data_low)
    print(data_low.shape)

df['filter'] = data
df['filter'] = df['filter'].apply(repeat_to_length, length=max_length)
df.head()
del data,data_hig,data_low

#%%
x = np.stack(df['filter'].values, axis=0)
y = np.stack(df['label'].values, axis=0)




# %%
x_train, x_test, y_train, y_test, train_filenames, test_filenames = train_test_split(x, y,df['path'].values, train_size = 0.7, test_size=0.3)
#print("x_train: {0}, x_test: {1}".format(x_train.shape, x_test.shape))
print(x_train)
del x,y
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