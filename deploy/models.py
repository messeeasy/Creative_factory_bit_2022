import torch
import torch.nn as nn
import torch.nn.functional as F

# in_channel = 1
class CNN_conv1D(nn.Module):
    def __init__(self, in_channel, filter_num, filter_size, strides, pool_strides, dropout_para):
        super(CNN_conv1D, self).__init__()
        self.model = nn.Sequential(

            nn.Conv1d(in_channel, filter_num[0], filter_size[0], stride=strides[0]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[0], pool_strides[0]),
            nn.BatchNorm1d(filter_num[0]),
            nn.Conv1d(filter_num[0], filter_num[1], filter_size[1], stride=strides[1]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[1], pool_strides[1]),
            nn.BatchNorm1d(filter_num[1]),
            nn.Conv1d(filter_num[1], filter_num[2], filter_size[2], stride=strides[2]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[2], pool_strides[2]),
            nn.BatchNorm1d(filter_num[2]),
            nn.Conv1d(filter_num[2], filter_num[3], filter_size[3], stride=strides[3]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[3], pool_strides[3]),
            nn.BatchNorm1d(filter_num[3]),
            nn.Dropout(dropout_para[0]),
            nn.Conv1d(filter_num[3], filter_num[4], filter_size[4], stride=strides[4]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[4], pool_strides[4]),
            nn.BatchNorm1d(filter_num[4]),
            nn.Dropout(dropout_para[1]),
            nn.Conv1d(filter_num[4], filter_num[5], filter_size[5], stride=strides[5]),
            nn.ReLU(),
            nn.MaxPool1d(filter_size[5], pool_strides[5]),
            nn.BatchNorm1d(filter_num[5]),
            nn.Dropout(dropout_para[2]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_num[5],2)

        )

    def forward(self, x):
        x = self.model(x)
        return x


class CNN_conv2D(nn.Module):
    def __init__(self, in_channel, filter_num, filter_size, strides, pool_strides, dropout_para):
        super(CNN_conv2D, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(in_channel, filter_num[0], (1,filter_size[0]), stride=strides[0]),
            nn.ReLU(),
            nn.MaxPool2d((1,filter_size[0]), pool_strides[0]),
            nn.BatchNorm2d(filter_num[0]),
            nn.Conv2d(filter_num[0], filter_num[1], (1,filter_size[1]), stride=strides[1]),
            nn.ReLU(),
            nn.MaxPool2d((1,filter_size[1]), pool_strides[1]),
            nn.BatchNorm2d(filter_num[1]),
            nn.Conv2d(filter_num[1], filter_num[2], (1,filter_size[2]), stride=strides[2]),
            nn.ReLU(),
            nn.MaxPool2d((1,filter_size[2]), pool_strides[2]),
            nn.BatchNorm2d(filter_num[2]),
            nn.Conv2d(filter_num[2], filter_num[3], (1,filter_size[3]), stride=strides[3]),
            nn.ReLU(),
            nn.MaxPool2d((1, filter_size[3]), pool_strides[3]),
            nn.BatchNorm2d(filter_num[3]),
            nn.Dropout(dropout_para[0]),
            nn.Conv2d(filter_num[3], filter_num[4], (1,filter_size[4]), stride=strides[4]),
            nn.ReLU(),
            nn.MaxPool2d((1,filter_size[4]), pool_strides[4]),
            nn.BatchNorm2d(filter_num[4]),
            nn.Dropout(dropout_para[1]),
            nn.Conv2d(filter_num[4], filter_num[5], (1,filter_size[5]), stride=strides[5]),
            nn.ReLU(),
            nn.MaxPool2d((1,filter_size[5]), pool_strides[5]),
            nn.BatchNorm2d(filter_num[5]),
            nn.Dropout(dropout_para[2]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(filter_num[5],2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


# input_shape=(N, L, C)?
# 入力参照 : https://qiita.com/sloth-hobby/items/93982c79a70b452b2e0a, https://aidiary.hatenablog.com/entry/20180902/1535887735

class LSTM_conv1D(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, hidden_size2, num_layer, dropout):
        super(LSTM_conv1D, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layer = num_layer,
                            batch_first = True,
                            dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size2, output_size)

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc1(x[:, -1])
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x
        
        

        


