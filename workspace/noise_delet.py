import torch

def standard_deviation(data):
    data = torch.FloatTensor(data)
    data_mean=torch.mean(data)
    data_std=torch.std(data)
    print(data_std)
    print(data_mean)

    for i in range(len(data)):
        if abs(data[i])<abs(data_std):
            #data[i]/=data_std
            data[i]=0
    return data
