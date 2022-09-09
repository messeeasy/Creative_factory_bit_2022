# Creative_factory_bit_2022

## Content
---
・Outline  
・Execution Method  
・Result  
・Current problems  
・Solution  
・Future plans  
・Information  
<br>

## Outline
---
We located some Github code on heart beat classification and demonstrated it using our dataset.  
Link:  
・[Frequency Conversion Functions](https://github.com/nicolaxs69/Phonocardiogram_Processing)  
・[Heart Sound Classification sample 1](https://github.com/aptr288/Heart_Sound_Classification)  
・[Heart Sound Classification sample 2](https://github.com/18D070001/Heart_sound_classification)  
  
Below is the current progress (will be updated as needed)  
<br>
・SVM, MLP, CNN based model training  
・LSTM-based training (using mcff features)  
・CNN-based training (using mel-spectrogram features)  
・Frequency Analysis  
・Learning with SVM, MLP, and CNN-based models on frequency-converted data  
・Learning with 2D CNN-based models  
<br>
Each file name, etc., should be modified for clarity. 
<br>

## Execution Method 
---
The Dataset should be placed in the root directory as 'dataset_heart_sound'.
Please refer to the following for the contents of 'dataset_heart_sound'. <br>
Creative_factory_bit_2022<br>
|-- dataset_heart_sound/<br>
|	|-- AV/<br>
| |	|-- abnormal/<br>
| |	'-- normal/<br>
| |-- MV/<br>
| |	|-- abnormal/<br>
| |	'-- normal/<br>
| |-- PV/<br>
| |	|-- abnormal/<br>
| |	'-- normal/<br>
| |-- TV/<br>
| |	|-- abnormal/<br>
| |	'-- normal/<br>
| |-- abnormal/<br>
| '-- normal/<br>
'-- workspace/<br>

See requirements.txt for the libraries currently in use.<br>
### Correspondence table for each file (to be added later)<br>
| File name | Detail | 
| :---------:| :------------------ |
| data_arrange.py | 音声ファイルの読み込みやサイズ調整 | 
| data_augment.py | pytorchでのデータセットを作る<br>ホワイトノイズデータセット、シフトデータセットの作成<br>特徴量変換| 
| FC_function.py | フーリエ変換 |
| generate_param.py | ハイパーパラメータの組み合わせを生成 |
| k_fold.py | TrainデータセットをK分割 |  
| models.py | 学習モデルの呼び出し |  
| noise_delete.py | ノイズ除去 | 
| plot.py | 学習結果の保存・出力 |
| self_adjust_noise.py | ノイズ除去のパラーメタ調整＆効果を確認 | 
| train.py | 学習パラメータの設定 |  
<br>

The CNN_demo folder is the ipynb file in Audio Classification ANN CNN Keras/References is the demo file.  
<br>

## Result
---
We trained a demo file (SVM, MLP, CNN_conv1D) of heartbeat classification that we found on GIthub on our dataset.
In this training, the training data is still a signal, so we performed a Fourier transform.
To change the convolution method, we transformed 0~1000Hz of the Fourier transformed data to (10*100) and trained with a 2-dimensional CNN.
Since the Fourier transform does not include time series, we trained a 2-D CNN using the mel-spectrogram, which uses the short-time Fourier transform, as the feature.<br>

Training with Train is highly accurate, but training with val,Test is not stable, 40~60%. 
Overlearning is occurring.<br>
![accuracy_20220909_044808](https://user-images.githubusercontent.com/52558553/189246781-83731220-734b-42cc-bb98-6f71b4768a14.png)
![loss_20220909_044808](https://user-images.githubusercontent.com/52558553/189246998-fab8e099-f70f-4e9e-95af-619e1cca226e.png)
![test_confusion_20220909_044808](https://user-images.githubusercontent.com/52558553/189246807-e6d91e99-3b89-4aa6-94c6-f937ea8c1288.png)
![ROCcurve_20220909_033001](https://user-images.githubusercontent.com/52558553/189246738-fc31f443-a2e7-4c44-97fa-9c197dfe196d.png)
<br>

When the executable file is run, the result is output as an image file in the output folder.<br>

### Data of heart sound
A plot of the heart sound data for the current data set shows the following variation.<br>
![Screenshot from 2022-08-19 11-42-26](https://user-images.githubusercontent.com/52558553/187862288-c509ddaa-35cb-490a-be8a-abfcd6a65d64.png)
![Screenshot from 2022-08-19 11-44-07](https://user-images.githubusercontent.com/52558553/187862311-51a80084-e7c5-4da5-976c-1035ee6003ea.png)
![Screenshot from 2022-08-19 11-45-11](https://user-images.githubusercontent.com/52558553/187862326-5229c973-eba3-4a2d-a4c5-e61dea5d0e58.png)

The frequency of heart sounds is less than 1 kHz, and the main component is about 100 Hz, while heart murmurs are relatively high frequency and are often found above 200 Hz compared to I and II sounds. Therefore, frequency analysis is used to analyze abnormal heart sounds.  <br>
[Ref 1: Sorry for Japanese. I think you can find it if you look for it.](https://www.cst.nihon-u.ac.jp/research/gakujutu/53/pdf/M-20.pdf)  <br>
 
Noise removed.<br>
The mean and standard deviation of each signal were determined, and values greater than N (currently N=4) times the standard deviation were replaced with the mean value.
<br>

Example of Noise Removal by Standard Deviation (N = 1) :  
・Green is standard deviation  
・Red is the mean  
・Blue is original data  
・Orange is after removal  

![Screenshot from 2022-09-09 09-18-09](https://user-images.githubusercontent.com/52558553/189250422-5bf99322-f32a-4fbb-a502-fff10fe48823.png)
![Screenshot from 2022-09-09 09-18-23](https://user-images.githubusercontent.com/52558553/189247431-56cf1b4e-7483-4ce9-9864-4d785f84d96c.png)

A low-pass filter was used to set the pass-end frequency above 120 Hz, the stop-end frequency above 500 Hz, the maximum loss at the passband end at 5 dB, and the minimum loss at the stopband end at 20 dB.<br>  

Noise reduction sample: normal/50214.wav
![Screenshot from 2022-09-09 09-36-35](https://user-images.githubusercontent.com/52558553/189249048-988a2bf5-fae8-4520-9ee1-6ae6133af7a2.png)
![Screenshot from 2022-09-09 09-39-19](https://user-images.githubusercontent.com/52558553/189249050-49c46211-add0-4b4b-9712-60f7d6558b38.png)
<br>
However, it was not quite accurate.
Looking at other data, we found the following data.
These files do not sound heartbeat properly.
<br><br>
Bad sample: normal/49653.wav
<br>

![Screenshot from 2022-09-09 09-46-31](https://user-images.githubusercontent.com/52558553/189249658-d56e7230-eabb-4597-8682-03af1df2342e.png)

<br>

## Current problem
---
### 1. Hyperparameters and multiple methods/models do not improve verification accuracy.
Over-learning is occurring. It is not stable.
### 2. malicious files exist in the dataset.
<br>
When the audio file is checked, it is the sound of breathing, or in some files, the sound of a baby's cry.  
<br><br>

## Solution
---
### 1.Exclude files from the data set that cannot be removed by noise reduction
### 2.Further adjustment of hyperparameters
<br>

## Future plans
---
<br>

1. hyper-parameter adjustment
2. image-based classification
3. removal of bad data
4. summary of results  

<br>


## Information
---
Sorry for the delay in working on this, but I will have a summary of the overall results up on github by 9/11.