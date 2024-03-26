import torch
import torch.nn as nn
import numpy as np
from tensorly.decomposition import partial_tucker
import tensorly
import VBMF

def conv_weight_mat_tucker_decomposition(tensor):
    '''之后考虑动态寻找最优的主成分个数'''
    rank = 16

    core, factors = partial_tucker(tensor.numpy(), modes=[1], rank=[rank], init='svd')
    return core, factors

tensor = torch.randn(64,32,3,3)

core,factors = conv_weight_mat_tucker_decomposition(tensor)
print(core.shape)
print(factors[0].shape)