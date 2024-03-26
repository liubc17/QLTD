from calculate_model_size.model_size import compute_model_nbits
import torch
from lowrank_compress_R50 import lowrank_compress_model
# from lowrank_compress_V16 import lowrank_compress_model
from torch import nn
from typing import Callable, Dict, List, NewType, Optional, Set, Union
from config.config_loader import load_config
import os
from permutation import permute_model
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from quantization_compress_utils.quantization_compress import quantization_compress_model
from torchvision.models.resnet import model_urls

# def get_resnet18(pretrained: bool = False) -> torch.nn.Module:
#     """Get PyTorch's default ResNet-18 model"""
#     # Hack to fix SSL error while loading pretrained model -- see https://github.com/pytorch/pytorch/issues/2271
#     model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
#     model = torchvision.models.resnet18(pretrained=pretrained)
#     model._arch = "resnet18"
#     return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""load config"""
file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config/train_resnet18_ILSVRC2012_noshortcut.yaml")
config = load_config(file_path, default_config_path=default_config)
model_config = config["model"]
compression_config = model_config["compression_parameters"]

'''get pretrained_model'''
model = torch.load('save_model/origin_model/Resnet18_pretrained_imagenet.pt')
# print(model)
# model = get_resnet18(True)
print('原模型',compute_model_nbits(model))
origin_bits = compute_model_nbits(model)

'''lowrank-factorization'''
lowrank_compressed_model = lowrank_compress_model(model)
lowrank_compressed_model = lowrank_compressed_model.cuda()
# # #

print('低秩后',compute_model_nbits(lowrank_compressed_model))
# torch.save(lowrank_compressed_model,'save_model/VGG16/lowrank/lowrank_4_compressed_model_imagenet.pt')
# print(lowrank_compressed_model)
# print('压缩比',origin_bits/compute_model_nbits(lowrank_compressed_model))

# lowrank_compressed_model = torch.load('save_model/VGG16/lowrank/lowrank_2_compressed_model_imagenet_9ep_trained.pt')
# print(lowrank_compressed_model)
# lowrank_bits = compute_model_nbits(lowrank_compressed_model)
# # print(lowrank_compressed_model)
# print('压缩比',origin_bits/lowrank_bits)
# torch.save(lowrank_compressed_model,'save_model/R50/lowrank/lowrank_4_compressed_model_imagenet.pt')

'''permutation'''
if "permutations" in model_config and model_config.get("use_permutations", False):
    permute_model(
        model,
        compression_config["fc_subvector_size"],
        compression_config["pw_subvector_size"],
        compression_config["large_subvectors"],
        permutation_groups=model_config.get("permutations", []),
        layer_specs=compression_config["layer_specs"],
        sls_iterations=model_config["sls_iterations"],
    )

quantization_compress_model = quantization_compress_model(model, **compression_config).cuda()
# print(quantization_compress_model)
quantization_bits = compute_model_nbits(quantization_compress_model)
print('低秩+量化后',quantization_bits)
print('压缩比',origin_bits/quantization_bits)
torch.save(quantization_compress_model,
           'save_decom_2_2_0.01_R18_ImageNet.pt')

# #
#
#
#
#















#
# # '''可视化网络模型'''
# # input_data = Variable(torch.rand(16,3,224,224))
# # input_data = input_data.to(device)
# # writer = SummaryWriter(logdir="visual/log",comment='Resnet18_lowrank')
# # with writer:
# #     writer.add_graph(model, (input_data))
# print('原模型',compute_model_nbits(model))
