# Creative_factory_bit_2022

## 内容
---
・概略  
・アルゴリズム&手法  
・実行方法    
・結果  
<br>

## 概要
---
私達は心音の分類に関するいくつかのGithubコードを探し、私達のデータセットを利用してデモを行いました。  
リンク:  
・[Frequency Conversion Functions](https://github.com/nicolaxs69/Phonocardiogram_Processing)  
・[Heart Sound Classification sample 1](https://github.com/aptr288/Heart_Sound_Classification)  
・[Heart Sound Classification sample 2](https://github.com/18D070001/Heart_sound_classification)  
  
私達は心音データを以下の方法で学習させました。<br>
・信号データをSVM,MLP,CNN_1Dにて学習  
・フーリエ変換を施したデータをSVM,MLP,CNN_1Dにて学習<br>
・メルスペクトログラムに変換し、CNN＿2Dにて学習<br>
・メルスペクトログラムの画像データをResNet50にて学習<br>
・MFCCに変換し、LSTMにて学習<br>

## アルゴリズム&手法
---
![messageImage_1662798286401](https://user-images.githubusercontent.com/52558553/189476577-52dbd23d-a18a-4fa5-a48f-1880c717f2e1.jpg)

まず、各信号データのノイズを除去し、ホワイトノイズデータセットとシフトしたデータセットを生成し、Data augmentationを行った。<br>
これらのデータセットに対して、そのまま、フーリエ変換、メルスペクトグラム、MFCCを用いて学習を行った。<br>
学習に使用したモデルは上のフローチャートに対応したものを使用している。
<br>
<br>

## 実行方法
---
Datasetは'dataset_heart_sound'としてルートディレクトリに置いてください。
'dataset_heart_sound'の構成内容は以下を参考にしてください。

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

現在使用しているライブラリはrequirements.txtを参照してください。<br><br>
### 各ファイルの対応表 <br>
| ファイル名 | 概要 | 
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

それぞれのファイル名などは分かりやすく修正しておきます。<br>
私達が作成したファイルはdeployフォルダに入っています。<br>
_exe.pyは各手法での実行ファイルです。<br>
_sample.pyは各関数のテストファイルです。<br>
その他は関数ファイルです。<br>
CNN_demoフォルダはAudio Classification ANN CNN Keras/References内のipynbファイルがデモファイルです。<br>
<br>

<br>

## 結果
---
### Story
GIthub上にあった心音の分類のデモファイル(SVM,MLP,CNN_conv1D)を私たちのデータセットで学習した。
この学習では学習データは信号のままなので、フーリエ変換を行った。
畳み込みのやり方を変えるため、フーリエ変換をしたデータの0~1000Hzを(10*100)に変換し、２次元のCNNで学習した。
フーリエ変換では時系列は含まれていないので、短時間フーリエ変換を用いるメルスペクトグラムを特徴量として使用した2次元のCNNの学習を行った。  
メルスペクトログラムをMATLABを使って画像データとして保存し、ResNet50で学習させた。  
また、特徴量としてMCFFを使用したLSTMでの学習を行った。


Trainでの学習は高精度だが、val,Testでの学習が安定しない、4~6割。過学習を起こしている。  
過学習を防ぐため、Dropoutやレイヤーのパラメータの調節を行ったが、その際はTrainも精度が下がってしまった。

| Methods | K平均のTrain acc | K平均のTest acc | AUC |
| :---------:| :------------------: | :------------------: | :------------------: |
| 信号のSVM| 4 | 5 | 6|
| 信号のMLP | | | |
| 信号のCNN 1D |  | | |
| フーリエ変換のSVM |  | | |
| フーリエ変換のMLP |  |  | |
| フーリエ変換のCNN 1D |  | | |
| LSTM & mcff　|  |  | |
| CNN & mel | | | |
| ResNet & mel |  | | |

混同行列などの出力はfinal_outputフォルダに含まれています。

<br>

実行ファイルを実行するとoutputフォルダに結果が画像ファイルで出力されます。

サンプル：　CNN_2D_melspectの結果例　他のものも似たような結果になる
![accuracy_20220909_044808](https://user-images.githubusercontent.com/52558553/189246781-83731220-734b-42cc-bb98-6f71b4768a14.png)
![loss_20220909_044808](https://user-images.githubusercontent.com/52558553/189246998-fab8e099-f70f-4e9e-95af-619e1cca226e.png)
![test_confusion_20220909_044808](https://user-images.githubusercontent.com/52558553/189246807-e6d91e99-3b89-4aa6-94c6-f937ea8c1288.png)
![ROCcurve_20220909_033001](https://user-images.githubusercontent.com/52558553/189246738-fc31f443-a2e7-4c44-97fa-9c197dfe196d.png)

### 心音データ
今回のデータセットの心音データをプロットすると以下のようにばらつきがあることが分かります。<br>
![Screenshot from 2022-08-19 11-42-26](https://user-images.githubusercontent.com/52558553/187862288-c509ddaa-35cb-490a-be8a-abfcd6a65d64.png)
![Screenshot from 2022-08-19 11-44-07](https://user-images.githubusercontent.com/52558553/187862311-51a80084-e7c5-4da5-976c-1035ee6003ea.png)
![Screenshot from 2022-08-19 11-45-11](https://user-images.githubusercontent.com/52558553/187862326-5229c973-eba3-4a2d-a4c5-e61dea5d0e58.png)

心音の周波数は1kHz以下，主成分は100Hz程度であり，心雑音はI音，II音に比べて比較的高周波で200Hz以上に見られることが多いです。そのため、周波数解析により、心音の異常解析を行う。  
[Ref 1 日本語でごめんなさい。探せばあると思います。](https://www.cst.nihon-u.ac.jp/research/gakujutu/53/pdf/M-20.pdf)  

ノイズを除去しました。
各信号の平均と標準偏差を求め、標準偏差のN(現在N=4)倍よりも大きい値を平均値に置き換える。
  
<br>
標準偏差によるノイズ除去の例　（N = 1)  
・緑が標準偏差  
・赤が平均  
・青が元データ  
・オレンジが除去後  

![Screenshot from 2022-09-09 09-18-09](https://user-images.githubusercontent.com/52558553/189250422-5bf99322-f32a-4fbb-a502-fff10fe48823.png)
![Screenshot from 2022-09-09 09-18-23](https://user-images.githubusercontent.com/52558553/189247431-56cf1b4e-7483-4ce9-9864-4d785f84d96c.png)

ローパスフィルタを使って、120Hz以上を通過端周波数、500Hz以上を阻止端周波数、通過域端最大損失は5dB、阻止域端最小損失20dBに設定しました。  

ノイズ除去のサンプル: normal/50214.wav  
![Screenshot from 2022-09-09 09-36-35](https://user-images.githubusercontent.com/52558553/189249048-988a2bf5-fae8-4520-9ee1-6ae6133af7a2.png)  
![Screenshot from 2022-09-09 09-39-19](https://user-images.githubusercontent.com/52558553/189249050-49c46211-add0-4b4b-9712-60f7d6558b38.png)
  
しかし、いまいち精度が上がりませんでした。
他のデータを見ると以下のようなデータも見つかりました。
これらのファイルは心音がまともに聞こえません。
<br>
悪いサンプル: normal/49631.wav

![Screenshot from 2022-09-10 17-58-12](https://user-images.githubusercontent.com/52558553/189476458-8230d30e-81b2-4e63-92a0-9b95a702b588.png)


<br>





