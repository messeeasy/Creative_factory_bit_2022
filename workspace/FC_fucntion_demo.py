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

#%%

Fs1, data1 = wf.read("../dataset_heart_sound/AV/abnormal/9979_AV.wav")
print(data1)

data1 = FC_fucntion.vec_nor(data1)
pcgFFT1, vTfft1 = FC_fucntion.fft_k(data1, Fs1, 2000)

E_PCG,C = FC_fucntion.E_VS_100(pcgFFT1, vTfft1, 'percentage')

print(E_PCG[0])
#%%
# Showing Results in Pandas
#data = {'N1': np.round(E_PCG )}
#print(data)
print('Registros Patologicos')
df=pd.DataFrame(np.round(E_PCG ),index=['Total (%)','0-100Hz','100-200Hz','2000-300Hz','300-400Hz','400-500Hz','500-600Hz','600-700Hz','700-800Hz','800-900Hz','900-1000Hz'],columns=['P1'])
print (df)

#%%
print(E_PCG[0])
#%%

plt.figure(1)

plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(vTfft1[C[0]:C[10]], pcgFFT1[C[0]:C[10]],'r')
plt.show()

hz_list = ['0-100Hz','100-200Hz','2000-300Hz','300-400Hz','400-500Hz','500-600Hz','600-700Hz','700-800Hz','800-900Hz','900-1000Hz']
for i in range(len(C)-1):
    plt.subplot(5,1,1)
    plt.title('Transformada de Fourier '+ hz_list[i])
    plt.plot(vTfft1[C[i]:C[i+1]], pcgFFT1[C[i]:C[i+1]],'r')
    plt.show()
    
print(C)
print(len(vTfft1))
print(len(pcgFFT1))

"""
plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(E_PCG[1[0]] : E_PCG[1], pcgFFT1,'r')
"""
# %%
