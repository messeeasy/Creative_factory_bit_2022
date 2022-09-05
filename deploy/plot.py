# 学習ログ解析
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import datetime

def evaluate_history(history):
    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    now = datetime.datetime.now()
    filename = './output/loss_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    filename = './output/accuracy_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)

    return now
    
def test_result(net, test_loader, now, device):
    # テストデータでの検証
    label_list = []
    pred_list = []
    
    for (inputs, labels) in test_loader:
        # 正しいか不明
        inputs = inputs.to(device)
        labels = labels.numpy().tolist()
        pred = net(inputs).max(1).gpu().numpy().tolist()
        label_list += labels
        pred_list += pred
        
    print(accuracy_score(label_list, pred_list))
    
    # あとで、感度・特異度になるように入れ替える
    tag = ['abnormal', 'normal']
    cm = confusion_matrix(label_list, pred_list, labels = tag)
    df_cm = pd.DataFrame(cm, index=tag, columns=tag)
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_cm, annot=True, cbar=False, square=True, fmt="d")
    plt.xlabel("Predict labels")
    plt.ylabel("Test labels")
    plt.title("Confusion matrix")
    filename = './output/test_confusion_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.show(filename)