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
・SVM, MLP, CNNベースモデルでの学習  
・LSTMベースでの学習（mcffの特徴量を使用）  
・CNNベースにメルスペクトログラムの特徴量で学習  
・周波数分析  
・周波数に変換したデータでのSVM, MLP, CNNベースモデルでの学習  
・CNNベースモデルの2D化での学習  
<br>
それぞれのファイル名などは分かりやすく修正しておきます。  
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

現在使用しているライブラリはrequirements.txtを参照してください。<br>
### 各ファイルの対応表 （後に追記します。）<br>
| ファイル名 | 概要 | 
| :---------:| :------------------ |
| `` | Demo | 
| `` | Demo | 
<br>

CNN_demoフォルダはAudio Classification ANN CNN Keras/References内のipynbファイルがデモファイルです。  
<br>

## 結果
---
### 心音データ
今回のデータセットの心音データをプロットすると以下のようにばらつきがあることが分かります。<br>
![Screenshot from 2022-08-19 11-42-26](https://user-images.githubusercontent.com/52558553/187862288-c509ddaa-35cb-490a-be8a-abfcd6a65d64.png)
![Screenshot from 2022-08-19 11-44-07](https://user-images.githubusercontent.com/52558553/187862311-51a80084-e7c5-4da5-976c-1035ee6003ea.png)
![Screenshot from 2022-08-19 11-45-11](https://user-images.githubusercontent.com/52558553/187862326-5229c973-eba3-4a2d-a4c5-e61dea5d0e58.png)

心音の周波数は1kHz以下，主成分は100Hz程度であり，心雑音はI音，II音に比べて比較的高周波で200Hz以上に見られることが多いです。そのため、周波数解析により、心音の異常解析を行う。  
[Ref 1 日本語でごめんなさい。探せばあると思います。](https://www.cst.nihon-u.ac.jp/research/gakujutu/53/pdf/M-20.pdf)  
 

これらのデータを以下で学習させましたが、いずれもテストでの精度は低く、不安定です。  
<br>
・SVM, MLP, CNNベースモデルでの学習  
・LSTMベースでの学習（mcffの特徴量を使用）  
・CNNベースにメルスペクトログラムの特徴量で学習  
・周波数に変換したデータでのSVM, MLP, CNNベースモデルでの学習  
・CNNベースモデルの2D化での学習  

例:  
![Screenshot from 2022-09-01 17-05-31](https://user-images.githubusercontent.com/52558553/187864502-3b8052d3-30ad-4a58-b3b8-cdd795c72446.png)
<br>

## 現在の問題点
---
### 1. データのサイズが長過ぎる  
１つの心音ファイルに何十回も心音が聞こえるので、メモリ的にも学習的にも分割するほうが良い
### 2. ノイズが多い  
音声ファイルを確認すると呼吸の音だったり、ファイルによっては赤ちゃんの鳴き声が含まれていたりする。  
<br>

## 解決案
---

### 1.1 データを分割して、交差検証法を行う
### 2.1 信号の標準偏差を求めて、標準偏差*2 or 3の値を持つ部分を平均に置き換える
### 2.2 ハイパスフィルターやローパスフィルターを使用する
<br>

## 今後のプラン
---
<br>

1. データの分割  
2. ベースとなるノイズ除去  
3. ファイルの整理  
4. K-fold cross validation  
5. 特徴量を追加・変更  
6. モデルの層の調整  
7. ハイパーパラメータの調整  
8. LSTM 2Dモデルの作成　(余裕があれば)  
9. プロット画像ベースの分類  （余裕があれば）  
<br>

| Schedule | 9/1 | 9/2 | 9/3 | 9/4 | 9/5 | 9/6 | 9/7 | 9/8 | 9/9 ~ |  
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------|:-----------:|:-----------:|:-----------:|:-----------|:-----------|  
| Task number | 1, 2 | 2, 3 | 3, 4 | 5, 6, 7 | 6, 7 | 8, 9 | 9 | 9 | 7 |  
<br>

## 諸連絡
---
現在、飯島、山本の二人でプロジェクトを進めています。  
また、飯島が先週インターンに参加していたため、進んでいません。  
9/8まで集中してやるので、結果はぎりぎりになります。すいません。  
