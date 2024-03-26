import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorly.decomposition import partial_tucker
from quantization_compress_utils.CompressedConv2d import CompressedConv2d


def conv_weight_mat_tucker_decomposition(tensor):
    '''之后考虑动态寻找最优的主成分个数'''
    size_list = list(tensor.size())
    '''只对output_channels和input_channels这两个维度进行tucker'''
    size_list[0] = int(size_list[0] / 4)
    size_list[1] = int(size_list[1] / 4)
    core, factors = partial_tucker(tensor.numpy(), modes=[0, 1], rank=[size_list[0], size_list[1]], init='svd')
    return core, factors


# 获取稀疏张量和mask
def gain_sparse_tensor_mask(weights, sparse_ratio, mode: str = 'soft', return_mask: bool = True):
    if isinstance(weights, torch.Tensor):
        weights = np.array(weights.cpu())
    if isinstance(weights, np.ndarray):
        # 获得阈值
        col_vector = sorted(np.abs(weights.ravel()))
        threshold_value_idx = round(len(col_vector) * (1 - sparse_ratio))
        threshold_value = col_vector[threshold_value_idx]
        if mode == 'soft':
            s_weights = np.sign(weights) * np.maximum(np.abs(weights) - threshold_value, 0)
            mask = np.where(s_weights == 0, 0, 1)
        else:
            s_weights = np.where(np.abs(weights) < threshold_value, 0, weights)
            mask = np.where(s_weights == 0, 0, 1)
        if return_mask:
            return s_weights, mask
        else:
            return s_weights


class Tucker_first_conv_layer(nn.Module):
    def __init__(self, sparse_ratio, factor, ori_dilation):
        super(Tucker_first_conv_layer, self).__init__()
        self.sparse_ratio = sparse_ratio
        self.ori_dilation = ori_dilation
        self.factor = factor
        # pointwise,1*1卷积层
        self.first_layer = nn.Conv2d(in_channels=factor.shape[0],
                                     out_channels=factor.shape[1],
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     dilation=ori_dilation,
                                     bias=False)
        self.apply_weights()
        # 稀疏层参数
        with torch.no_grad():
            if isinstance(self.first_layer, CompressedConv2d):
                weights = self.first_layer._get_uncompressed_weight()
            if isinstance(self.first_layer, nn.Conv2d):
                weights = self.first_layer.weight.data
        s_weights, s_mask = gain_sparse_tensor_mask(weights, self.sparse_ratio)
        self.s_weights = nn.Parameter(torch.from_numpy(s_weights), requires_grad=True)
        self.s_mask = nn.Parameter(torch.from_numpy(s_mask), requires_grad=False)

    def apply_weights(self):
        factor = self.factor.transpose((1, 0))
        self.first_layer.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(factor.copy(), axis=-1), axis=-1)))

    def forward(self, x):
        device = x.device
        x1 = self.first_layer(x)
        x2 = F.conv2d(x, (self.s_weights*self.s_mask).to(device), stride=1, padding=0, dilation=self.ori_dilation)
        out = x1 + x2
        return out


class Tucker_core_conv_layer(nn.Module):

    def __init__(self, layer, sparse_ratio, tucker_core):
        super(Tucker_core_conv_layer, self).__init__()
        self.sparse_ratio = sparse_ratio
        self.core = tucker_core
        # D*D卷积层
        self.core_layer = nn.Conv2d(in_channels=tucker_core.shape[1],
                                    out_channels=tucker_core.shape[0],
                                    kernel_size=layer.kernel_size,
                                    stride=layer.stride,
                                    padding=layer.padding,
                                    dilation=layer.dilation,
                                    bias=False)
        self.stride = layer.stride
        self.padding = layer.padding
        self.apply_weights()
        # 稀疏层参数
        with torch.no_grad():
            if isinstance(self.core_layer, CompressedConv2d):
                weights = self.core_layer._get_uncompressed_weight()
            if isinstance(self.core_layer, nn.Conv2d):
                weights = self.core_layer.weight.data
        s_weights, s_mask = gain_sparse_tensor_mask(weights, self.sparse_ratio)
        self.s_weights = nn.Parameter(torch.from_numpy(s_weights), requires_grad=True)
        self.s_mask = nn.Parameter(torch.from_numpy(s_mask), requires_grad=False)

    def apply_weights(self):
        self.core_layer.weight.data = torch.from_numpy(np.float32(self.core.copy()))

    def forward(self, x):
        device = x.device
        x1 = self.core_layer(x)
        x2 = F.conv2d(x, (self.s_weights*self.s_mask).to(device), stride=self.stride, padding=self.padding)
        out = x1 + x2
        return out


class Tucker_last_conv_layer(nn.Module):

    def __init__(self, sparse_ratio, factor, bias, ori_dilation):
        super(Tucker_last_conv_layer, self).__init__()
        self.sparse_ratio = sparse_ratio
        self.factor = factor
        if bias is not None:
            b = bias
        else:
            b = False
        self.ori_dilation = ori_dilation
        # pointwise,1*1卷积层
        self.last_layer = nn.Conv2d(in_channels=factor.shape[1],
                                    out_channels=factor.shape[0],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=ori_dilation,
                                    bias=b)
        self.apply_weights()
        # 稀疏层参数
        with torch.no_grad():
            if isinstance(self.last_layer, CompressedConv2d):
                weights = self.last_layer._get_uncompressed_weight()
            if isinstance(self.last_layer, nn.Conv2d):
                weights = self.last_layer.weight.data
        s_weights, s_mask = gain_sparse_tensor_mask(weights, self.sparse_ratio)
        self.s_weights = nn.Parameter(torch.from_numpy(s_weights), requires_grad=True)
        self.s_mask = nn.Parameter(torch.from_numpy(s_mask), requires_grad=False)

    def apply_weights(self):
        self.last_layer.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(self.factor.copy(), axis=-1), axis=-1)))

    def forward(self, x):
        device = x.device
        x1 = self.last_layer(x)
        x2 = F.conv2d(x, (self.s_weights*self.s_mask).to(device), stride=1, padding=0, dilation=self.ori_dilation)
        out = x1 + x2
        return out


def tucker_conv_layer(layer: nn.Conv2d, sparse_ratio):
    core, factors = conv_weight_mat_tucker_decomposition(layer.weight.detach().cpu())
    bias = layer.bias
    first_conv_layer = Tucker_first_conv_layer(sparse_ratio, factor=factors[1], ori_dilation=layer.dilation)
    core_conv_layer = Tucker_core_conv_layer(layer, sparse_ratio, tucker_core=core)
    last_conv_layer = Tucker_last_conv_layer(sparse_ratio, factor=factors[0], bias=bias, ori_dilation=layer.dilation)
    new_layers = [first_conv_layer, core_conv_layer, last_conv_layer]
    return nn.Sequential(*new_layers)