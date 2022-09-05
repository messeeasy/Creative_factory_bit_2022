import torch
import torch.nn as nn
import numpy as np
import models
import torch.optim as optim

# 乱数固定　初期化
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn_deterministic = True
    torch.use_deterministic_algorithms = True

# CNN系モデルのインスタンス生成
def model_setting_cnn(model, in_channel ,filter_num, filter_size, strides, pool_strides, dropout_para, device):
    
    torch_seed()
    
    if model == 'CNN_conv1D':
        net = models.CNN_conv1D(in_channel ,filter_num, filter_size, strides, pool_strides, dropout_para)
        
    elif model == 'CNN_conv2D':
        net = models.CNN_conv2D(in_channel ,filter_num, filter_size, strides, pool_strides, dropout_para)
        
    net = net.to(device)
    
    return net

# LSTM系モデルのインスタンス生成
def model_setting_LSTM(model, input_size, output_size, hidden_size, hidden_size2, num_layer, dropout, device):    
    
    torch_seed()
    
    if model == 'LSTM_conv1D':
        net = models.LSTM_conv1D(input_size, output_size, hidden_size, hidden_size2, num_layer, dropout)
    
    net = net.to(device)
    
    return net

# model選択とここだけ呼び出せば良い
def training(net, lr, epoch, train_loader, val_loader, device):
    
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr= lr)
    
    history =  np.zeros((0,5))
    history, net = fit(net, optimizer, criterion, epoch, train_loader, val_loader, device, history)
    
    return history, net

# 損失計算用
def eval_loss(loader, device, net, criterion):

    # データローダーから最初の1セットを取得する
    for inputs, labels in loader:
        break

    # デバイスの割り当て
    inputs = inputs.to(device)
    labels = labels.to(device)

    # 予測計算
    outputs = net(inputs)

    #  損失計算
    loss = criterion(outputs, labels)

    return loss

# 学習用関数
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):

    # tqdmライブラリのインポート
    from tqdm.notebook import tqdm

    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        #訓練フェーズ
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        #予測フェーズ
        net.eval()
        count = 0

        for inputs, labels in test_loader:
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            val_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history, net

