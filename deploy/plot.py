# 学習ログ解析
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import datetime
import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def evaluate_history(history):
    #lossとaccuracyの確認
    print(f'First state: loss: {history[0,3]:.5f} accuracy: {history[0,4]:.5f}') 
    print(f'Final state: loss: {history[-1,3]:.5f} accuracy: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # Learning Curveの表示 (loss)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='Train')
    plt.plot(history[:,0], history[:,3], 'k', label='Val')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Learning Curve(loss)')
    plt.legend()
    now = datetime.datetime.now()
    filename = './output/loss_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)

    # Learning Curveの表示 (accuracy)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='Train')
    plt.plot(history[:,0], history[:,4], 'k', label='Val')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Learning Curve(accuracy)')
    plt.legend()
    filename = './output/accuracy_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)

    return now
    
def test_result(net, test_loader, now, device):
    # テストデータでのVal
    label_list = []
    pred_list = []
    pred_score = []
    
    for (inputs, labels) in test_loader:
        inputs = inputs.to(device)
        labels = labels.numpy().tolist()
        pred = torch.argmax(net(inputs),axis = 1).cpu().numpy().tolist()
        #pred_s = torch.max(net(inputs),dim = 1).values.item()
        label_list += labels
        pred_list += pred
        #pred_score.append(pred_s)
    
    print(len(test_loader))
    
    tag = ['normal', 'abnormal']
    label_tag = []
    pred_tag = []
    
    for label in label_list:
        if label == 0:
            label_tag.append('abnormal')
        else:
            label_tag.append('normal')
            
    for pred in pred_list:
        if pred == 0:
            pred_tag.append('abnormal')
        else:
            pred_tag.append('normal')
    
    cm = confusion_matrix(label_tag, pred_tag, labels = tag)
    df_cm = pd.DataFrame(cm, index=tag, columns=tag)
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_cm, annot=True, cbar=False, square=True, fmt="d")
    plt.xlabel("Predict labels")
    plt.ylabel("Test labels")
    plt.title("Confusion matrix")
    filename = './output/test_confusion_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)
    
    print(net(inputs).size())
    print(len(label_list))
    print(len(pred_score))

    return 0
"""
    fpr_all, tpr_all, thresholds_all = roc_curve(label_list, pred_score, drop_intermediate=False)
    
    plt.plot(fpr_all, tpr_all, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title("ROC curve")
    plt.grid()
    filename = './output/ROCcurve_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
    plt.savefig(filename)
    
    # あとで、テキストに出力する
    print('---- AUC score ----')
    print(roc_auc_score(label_list, pred_score))
"""

    # しきい値で変化させる