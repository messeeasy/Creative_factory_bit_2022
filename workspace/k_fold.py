import numpy as np

def k_fold(data,K):
    data_split= list(np.array_split(data, K, 0))
    return data_split