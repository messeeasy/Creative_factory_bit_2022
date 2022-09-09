# Creative_factory_bit_2022

## 内容
---
・概略
・実行方法    
・結果  
・現在の問題点  
・今後のプラン  
・諸連絡  
<br>

## 概要
---
私達は心音の分類に関するいくつかのGithubコードを探し、私達のデータセットを利用してデモを行いました。  
リンク:  
・[Frequency Conversion Functions](https://github.com/nicolaxs69/Phonocardiogram_Processing)  
・[Heart Sound Classification sample 1](https://github.com/aptr288/Heart_Sound_Classification)  
・[Heart Sound Classification sample 2](https://github.com/18D070001/Heart_sound_classification)  
  
以下は現状の進捗状況です　（随時更新します）  
<br>
・SVM, MLP, CNN（１次元）ベースモデルでの学習  
・CNNベースにメルスペクトログラムの特徴量で学習  
・周波数分析  
・フーリエ変換したデータでのSVM, MLP, CNNベースモデルでの学習  
・２次元でのCNNベースモデルにフーリエ変換を使用した学習  
・LSTMベースでの学習（mcffの特徴量を使用）  
・ノイズ除去  
・データセットのかさ増し　（ホワイトノイズデータとスライドさせたデータの追加）  
・２次元でのCNNベースモデルにメルスペクトグラムを使用した学習<br>


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

Trainでの学習は高精度だが、val,Testでの学習が安定しない、4~6割。過学習を起こしている。

| Methods | K平均のTrain acc | K平均のTest acc | AUC |
| :---------:| :------------------: | :------------------: | :------------------: |
| 信号のSVM|  |  | |
| 信号のMLP | | | |
| 信号のCNN 1D |  | | |
| フーリエ変換のSVM |  | | |
| フーリエ変換のMLP |  |  | |
| フーリエ変換のCNN 1D |  | | |

混同行列などの出力はoutputフォルダ

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
悪いサンプル: normal/49653.wav
![Screenshot from 2022-09-09 09-46-31](https://user-images.githubusercontent.com/52558553/189249658-d56e7230-eabb-4597-8682-03af1df2342e.png)


<br>

## 現在の問題点
---
### 1. ハイパーパラメータや複数の手法・モデルを使用しても検証精度が上がらない。
過学習を起こしている。安定しない。
### 2. データセットの中に悪質なファイルが存在する。
音声ファイルを確認すると呼吸の音だったり、ファイルによっては赤ちゃんの鳴き声が含まれていたりする。  


<br>

## 解決案
---

### 1.1 ノイズ除去では除去できないファイルをデータセットから除く
### 2.1 ハイパーパラメータの更なる調整

<br>

## 今後のプラン
---
<br>

1. ハイパーパラメータの調整

2. 画像ベースでの分類
3. 悪質なデータの削除
4. 結果のまとめ  

<br>

## 諸連絡
---
作業が遅れてすいません、総合的な結果をまとめたものを9/11までにgithubに上げておきます。
