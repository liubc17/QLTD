import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Tucker_layer import tucker_conv_layer
'''PS:不知道为什么我调用tucker_conv_layer分解测试用的卷积层会出问题'''

def get_sparse_tensor(layer_shape, sparse_ratio):
    # # 稀疏项
    shape = layer_shape  # [out,in,d,d]
    s_shape = (shape[0], shape[1]*shape[2]*shape[3])  # 稀疏层尺寸
    s_tensor, mask = sparse_matrix(s_shape, sparsity_ratio=sparse_ratio)

    s_weights = nn.Parameter(s_tensor.reshape(shape), requires_grad=True)
    mask = nn.Parameter(mask.reshape(shape), requires_grad=False)

    return s_weights, mask

def combined_layer(layer, sparse_ratio):
    return nn.Sequential(Combined_layer(layer, sparse_ratio))

# 指定稀疏比
def to_sparsity_ratio(matrix, sparsity_ratio):
    '''输入:矩阵,稀疏比
       输出:指定稀疏比的稀疏矩阵,类型为np.ndarray'''
    if isinstance(matrix, torch.Tensor):
        matrix = np.array(matrix)
    if isinstance(matrix, np.ndarray):
        # 获取breakpoint
        column_vector = matrix.ravel()
        sort_S = sorted(np.abs(column_vector))
        idx = round(len(sort_S) * (1-sparsity_ratio))
        breakpoint = sort_S[idx]
        # 软阈值处理
        new_sparse_matrix = np.sign(matrix)*np.maximum(np.abs(matrix)-breakpoint, 0)
        return new_sparse_matrix
    else:
        print('输入非np.ndarry,ERROR!,return None!')
        return None

# 获取稀疏比
# def get_sparsity_ratio(sparse_matrix):
#     total_elements = np.prod(sparse_matrix.shape)
#     nonzero_elements = np.count_nonzero(sparse_matrix)
#     sparsity_ratio = nonzero_elements/total_elements
#     return sparsity_ratio


def sparse_matrix(shape:tuple, sparsity_ratio):
    '''输入:尺寸,稀疏比
    输出:稀疏矩阵和mask掩码'''
    # 生成一组随机数
    random_matrix = np.random.randn(shape[0], shape[1])
    # 获取稀疏张量
    s_tensor = torch.Tensor(to_sparsity_ratio(random_matrix, sparsity_ratio))
    # 获取mask
    mask = torch.where(s_tensor == 0, torch.tensor(0), torch.tensor(1))
    # s_tensor = nn.Parameter(s_tensor,requires_grad=True)
    # mask = nn.Parameter(mask,requires_grad=False)
    return s_tensor, mask


'''稀疏层'''
class Sparse_layer(nn.Module):
    def __init__(self, layer_shape,sparse_ratio):
        super().__init__()
        self.sparse_layer = nn.Conv2d(layer_shape[1], layer_shape[0], kernel_size=1, stride=1, padding=0)
        self.s_tensor, self.mask = get_sparse_tensor(layer_shape, sparse_ratio=sparse_ratio)
    def forward(self, x):
        output = self.sparse_layer(x) * self.mask
        return output
# class Combined_layer(nn.Module):
#     def __init__(self,layer,sparse_ratio):
#         super().__init__()
#         self.tucker = tucker_conv_layer(layer)
#         layer_shape = layer.weight.shape
#         self.sparse = Sparse_layer(layer_shape,sparse_ratio=sparse_ratio)
#     def forward(self,x):
#         # x1 = self.tucker(x)
#         # x2 = self.sparse(x)
#         # out = x1 + x2
#         out = self.sparse(x)
#         return out


class Combined_layer(nn.Module):

    def __init__(self, layer, sparse_ratio):
        super().__init__()
        self.tucker = tucker_conv_layer(layer)
        layer_shape = layer.weight.shape
        self.stride = layer.stride[0],
        self.padding = layer.padding[0],
        self.dilation = layer.dilation[0],
        self.s_weights,self.mask = get_sparse_tensor(layer_shape, sparse_ratio=sparse_ratio)

    def forward(self, x):
        x1 = self.tucker(x)

        x2 = F.conv2d(x, self.s_weights*self.mask, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
        # print(self.s_weights.requires_grad)
        # print(self.mask.requires_grad)
        out = x2 + x1
        return out


if __name__ == '__main__':
    conv_layer = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    # layer_shape = conv_layer.weight.shape
    # print(layer_shape)
    test_net = Combined_layer(conv_layer, sparse_ratio=0.1)
    test_net2 = combined_layer(conv_layer, sparse_ratio=0.1)

    x = torch.randn((1, 3, 9, 9))
    y = test_net(x)
    print('done')
