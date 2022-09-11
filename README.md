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
   
<br>
We trained the heartbeat data with the following methods<br>
・Signal data is trained with SVM, MLP, and CNN_1D.  <br>
・Fourier transformed data is trained with SVM, MLP, and CNN_1D.<br>  
・Convert the Fourier transformed data into a Mel-Spectrogram and train with CNN_2D.<br>  
・Mel-spectrogram image data is trained with ResNet50.<br>
・Converted to MFCC and trained by LSTM.<br> 
<br>
<br>

## Algorithms & Methods
---
![messageImage_1662798286401](https://user-images.githubusercontent.com/52558553/189476577-52dbd23d-a18a-4fa5-a48f-1880c717f2e1.jpg)

First, noise was removed from each signal data set to generate a white noise data set and a shifted data set for data augmentation.<br>
Training was performed on these data sets as is, using Fourier transform, Mel-Spectrogram, and MFCC.<br>
The model used for training corresponds to the flowchart above.
<br>
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
### Correspondence table for each file <br>

| File name | Detail | 
| :---------:| :------------------ |
| data_arrange.py | Loading and sizing audio files | 
| data_augment.py | Make dataset using pytorch<br>Make White noise dataset and shift dataset<br>Convert feature| 
| FC_function.py | Convert to Fourier |
| generate_param.py | Generate hyperparameter combinations |
| k_fold.py | Train dataset divided into K parts |  
| models.py | Calling the learning model |  
| noise_delete.py | Delete noise | 
| plot.py | Save and output result about learning |
| self_adjust_noise.py | Adjust parrameters for noise reduction & check the effect | 
| train.py | Setting of learning parameters |  
<br>

Each file name, etc., should be modified for clarity.<br>
The files we created are in the deploy folder. <br>
_exe.py is the executable file for each method.<br>
_sample.py is the test file for each function.<br>
Others are function files.<br>
The CNN_demo folder is the ipynb file in Audio Classification ANN CNN Keras/References is the demo file.  
<br>

## Result
---
### Story

We trained a demo file (SVM, MLP, CNN_conv1D) of heartbeat classification that we found on Github on our dataset.  
In this training, the training data is still a signal, so we performed a Fourier transform.  
To change the convolution method, we transformed 0~1000Hz of the Fourier transformed data to (10*100) and trained with a 2-dimensional CNN.  
Since the Fourier transform does not include time series, we trained a 2-D CNN using the mel-spectrogram, which uses the short-time Fourier transform, as the feature.<br>

Training with Train is highly accurate, but training with val,Test is not stable, 40~60%. 
Overlearning is occurring.<br>

<br>

K = 5  
AV  
| Methods | Average(k) of Test acc | AUC |
| :---------:| :------------------: | :------------------: |
| SVM in signal | 0.541| 0.665|
| MLP in signal | 0.495| 0.622|
| CNN 1D in signal |   | |
| SVM in Fourier transform | 0.495 | 0.6278 |
| MLP in Fourier transform |  |   |
| CNN 1D in Fourier transform | 0.4965  | 0.49324|
| LSTM & mcff　|    | |
| CNN & mel | 0.55686 | 0.56582 |
| ResNet & mel | 0.53036 |0.51722 |
<br>

MV  
| Methods | Average(k) of Test acc | AUC |
| :---------:| :------------------: | :------------------: |
| SVM in signal | 0.5328 | 0.624|
| MLP in signal |0.504 | 0.656|
| CNN 1D in signal |   | |
| SVM in Fourier transform | 0.5184 | 0.5824 |
| MLP in Fourier transform |  |   |
| CNN 1D in Fourier transform |  0.5501 | 0.5117|
| LSTM & mcff　|    | |
| CNN & mel | 0.54326 | 0.59446 |
| ResNet & mel | 0.51722 | 0.52446|
<br>

PV  
| Methods | Average(k) of Test acc | AUC |
| :---------:| :------------------: | :------------------: |
| SVM in signal | 0.554 |0.606 |
| MLP in signal | 0.513| 0.590|
| CNN 1D in signal |   | |
| SVM in Fourier transform | 0.526 |  0.5636|
| MLP in Fourier transform |  |   |
| CNN 1D in Fourier transform |  0.4514 |0.5117 |
| LSTM & mcff　|    | |
| CNN & mel | 0.58058 | 0.50693 |
| ResNet & mel | 0.49806 | 0.45186|
<br>

TV  
| Methods | Average(k) of Test acc | AUC |
| :---------:| :------------------: | :------------------: |
| SVM in signal | 0.586 | 0.62|
| MLP in signal | | |
| CNN 1D in signal |   | |
| SVM in Fourier transform | 0.5202 | 0.5242 |
| MLP in Fourier transform |  |   |
| CNN 1D in Fourier transform |  0.52051 | 0.534022|
| LSTM & mcff　|    | |
| CNN & mel | 0.55047 | 0.56857 |
| ResNet & mel |0.54306  |0.56418 |
<br>

The output, including the confusion matrix, is contained in the final_output folder.  

When the executable is run, the result is output as an image file in the deploy/output folder.  
<br>

Sample: Example of CNN_2D_melspect results Others are similar<br>
![accuracy_20220909_044808](https://user-images.githubusercontent.com/52558553/189246781-83731220-734b-42cc-bb98-6f71b4768a14.png)
![loss_20220909_044808](https://user-images.githubusercontent.com/52558553/189246998-fab8e099-f70f-4e9e-95af-619e1cca226e.png)
![test_confusion_20220909_044808](https://user-images.githubusercontent.com/52558553/189246807-e6d91e99-3b89-4aa6-94c6-f937ea8c1288.png)
![ROCcurve_20220909_033001](https://user-images.githubusercontent.com/52558553/189246738-fc31f443-a2e7-4c44-97fa-9c197dfe196d.png)
<br>

When the executable file is run, the result is output as an image file in the output folder.<br>

### Data of heart sound
A plot of the heart sound data for the current data set shows the following variation.<br>
![Screenshot from 2022-08-19 11-42-26](https://user-images.githubusercontent.com/52558553/187862288-c509ddaa-35cb-490a-be8a-abfcd6a65d64.png)

The frequency of heart sounds is less than 1 kHz, and the main component is about 100 Hz, while heart murmurs are relatively high frequency and are often found above 200 Hz compared to I and II sounds. Therefore, frequency analysis is used to analyze abnormal heart sounds.  <br>
[Ref 1: Sorry for Japanese. I think you can find it if you look for it.](https://www.cst.nihon-u.ac.jp/research/gakujutu/53/pdf/M-20.pdf)  <br>
 
### Noise removed
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

