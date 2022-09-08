import numpy as np

def k_fold(data,y,K):
    data_split= list(np.array_split(data, K, 0))
    y_split= list(np.array_split(y, K, 0))
    return data_split,y_split