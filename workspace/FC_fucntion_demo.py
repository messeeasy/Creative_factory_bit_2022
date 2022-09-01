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
showFrequency=1000
Fs1, data1 = wf.read("../dataset_heart_sound/AV/abnormal/9979_AV.wav")
print(Fs1)

#data1 = FC_fucntion.vec_nor(data1)
pcgFFT1, vTfft1 = FC_fucntion.fft_k_N(data1, Fs1, showFrequency)
print(len(vTfft1))
plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(vTfft1, pcgFFT1,'r')
plt.show()
avg=np.zeros(showFrequency)
count=np.zeros(showFrequency)
 

for i in range(showFrequency-1):

    for j in range(len(vTfft1)):
            
        if i==0:
            if i<=vTfft1[j]and vTfft1[j]<i+1:
                avg[i]+=pcgFFT1[j]
                count[i]+=1
            elif i+1<=vTfft1[j]:
                break
        elif 0<i<showFrequency-1:
            if i-1<vTfft1[j]and vTfft1[j]<i+1:
                    avg[i]+=pcgFFT1[j]
                    count[i]+=1
            elif i+1<=vTfft1[j]:
                break
        else:
            if i-1<vTfft1[j]and vTfft1[j]<=i:
                avg[i]+=pcgFFT1[j]
                count[i]+=1
pcgFFT1_avg=avg/count
vTfft_avg=np.arange(0, 1000, 1)
print(len(vTfft_avg))
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#print(len(pcgFFT1_avg))
plt.figure(1)
plt.plot(vTfft_avg,pcgFFT1_avg)
plt.show()
#%%
E_PCG,C = FC_fucntion.E_VS_100_avg(pcgFFT1_avg, vTfft_avg, 'percentage')
#E_PCG,C = FC_fucntion.E_VS_100(pcgFFT1, vTfft1, 'percentage')
print(C)
#%%

plt.figure(1)

plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(vTfft_avg[C[0]:C[10]], pcgFFT1_avg[C[0]:C[10]],'r')
plt.show()

hz_list = ['0-100Hz','100-200Hz','2000-300Hz','300-400Hz','400-500Hz','500-600Hz','600-700Hz','700-800Hz','800-900Hz','900-1000Hz']
for i in range(len(C)-1):
    plt.subplot(5,1,1)
    plt.title('Transformada de Fourier '+ hz_list[i])
    plt.plot(vTfft_avg[C[i]:C[i+1]], pcgFFT1_avg[C[i]:C[i+1]],'r')
    plt.show()
    

print(len(vTfft_avg))
print(len(pcgFFT1_avg))

"""
plt.subplot(5,1,1)
plt.title('Transformada de Fourier')
plt.plot(E_PCG[1[0]] : E_PCG[1], pcgFFT1,'r')
"""
# %%
