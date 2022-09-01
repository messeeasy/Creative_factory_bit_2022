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
We located some Github code on heart　beat classification and demonstrated it using our dataset.　　
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
Please refer to the following for the contents of 'dataset_heart_sound'.  

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
| `` | Demo | 
| `` | Demo | 
<br>

The CNN_demo folder is the ipynb file in Audio Classification ANN CNN Keras/References is the demo file.  
<br>

## Result
---
### Data of heart sound
A plot of the heart sound data for the current data set shows the following variation.<br>
![Screenshot from 2022-08-19 11-42-26](https://user-images.githubusercontent.com/52558553/187862288-c509ddaa-35cb-490a-be8a-abfcd6a65d64.png)
![Screenshot from 2022-08-19 11-44-07](https://user-images.githubusercontent.com/52558553/187862311-51a80084-e7c5-4da5-976c-1035ee6003ea.png)
![Screenshot from 2022-08-19 11-45-11](https://user-images.githubusercontent.com/52558553/187862326-5229c973-eba3-4a2d-a4c5-e61dea5d0e58.png)

The frequency of heart sounds is less than 1 kHz, and the main component is about 100 Hz, while heart murmurs are relatively high frequency and are often found above 200 Hz compared to I and II sounds. Therefore, frequency analysis is used to analyze abnormal heart sounds.  
[Ref 1: Sorry for Japanese. I think you can find it if you look for it.](https://www.cst.nihon-u.ac.jp/research/gakujutu/53/pdf/M-20.pdf)  
 

### These data were trained on below, but the accuracy are low  and unstable.    
<br>
・SVM, MLP, CNN based model training<br>
・LSTM-based training (using mcff features)<br>  
・CNN-based training (using mel-spectrogram features)<br>  
・Training with SVM, MLP, and CNN-based models on frequency-converted data<br>  
・Learning with 2D CNN-based models<br>  
<br>

Example:  
![Screenshot from 2022-09-01 17-05-31](https://user-images.githubusercontent.com/52558553/187864502-3b8052d3-30ad-4a58-b3b8-cdd795c72446.png)
<br>

## Current problem
---
### 1. Data size is too long   
Dozens of heartbeats in one heartbeat file, so better to split them up for memory and learning purposes
### 2. Lots of noise  
When the audio file is checked, it is the sound of breathing, and some files contain the sound of a baby's cry.  
<br>

## Solution
---

### 1.1 Split data and perform cross-validation methods
### 2.1 Find the standard deviation of the signal and replace the time series with values of standard deviation*2 or 3 with the mean
### 2.2 Use a high-pass filter or low-pass filter
<br>

## Future plans
---
<br>

1. Data segmentation  
2. Noise removal  
3. File organization  
4. K-fold cross validation  
5. Adding or modifying features  
6. Adjusting model layers 
7. Adjust hyperparameters 
8. Create LSTM 2D model (if available) 
9. Plot-image-based classification (if available)  
<br>

| Schedule | 9/1 | 9/2 | 9/3 | 9/4 | 9/5 | 9/6 | 9/7 | 9/8 | 9/9 ~ |  
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|:-----------:|:-----------:|:-----------:|:-----------|:-----------|  
| Task number | 1, 2 | 2, 3 | 3, 4 | 5, 6, 7 | 6, 7 | 8, 9 | 9 | 9 | 7 |  
<br>

## Information
---
Currently, Iijima and Yamamoto are working on the project.  
Also, Iijima has not progressed since he participated in an internship last week.  
We will concentrate on the project until 9/8, so the result will be last minute.  
Sorry.