
from calculate_model_size.model_size import compute_model_nbits
import torch
from lowrank_compress import lowrank_compress_model
from torch import nn
from typing import Callable, Dict, List, NewType, Optional, Set, Union
from config.config_loader import load_config
import os
from permutation import permute_model
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from quantization_compress_utils.quantization_compress import quantization_compress_model
from resnetforcifar import resnet18cifar

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

"""load config"""
file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config/train_resnet18_cifar100_noshortcut.yaml")
config = load_config(file_path, default_config_path=default_config)
model_config = config["model"]
compression_config = model_config["compression_parameters"]

'''get pretrained_model'''

model = torch.load('save_model/origin_model/Resnet18_cifar100.pt')


print('原模型', compute_model_nbits(model))
origin_bits = compute_model_nbits(model)


lowrank_compressed_model = lowrank_compress_model(model)
# print(lowrank_compressed_model)
lowrank_compressed_model = lowrank_compressed_model.cuda()
# # #
lowrank_bits = compute_model_nbits(lowrank_compressed_model)
print('低秩后',lowrank_bits)

# for name, _ in lowrank_compressed_model.named_parameters():
#     print(name)
# torch.save_decom(lowrank_compressed_model,'save_model/LRQ/R18/lowrank/lowrank_4_compressed_model_cifar100.pt')
# print(lowrank_compressed_model)
# print('压缩比',origin_bits/lowrank_bits)
# lowrank_compressed_model = torch.load('save_model/LRQ/R18/lowrank/lowrank_4_compressed_model_cifar100.pt')

'''permutation'''
if "permutations" in model_config and model_config.get("use_permutations", False):
    print(11111)
    permute_model(
        lowrank_compressed_model,
        compression_config["fc_subvector_size"],
        compression_config["pw_subvector_size"],
        compression_config["large_subvectors"],
        permutation_groups=model_config.get("permutations", []),
        layer_specs=compression_config["layer_specs"],
        sls_iterations=model_config["sls_iterations"],
    )


quantization_compress_model = quantization_compress_model(lowrank_compressed_model, **compression_config).cuda()

quantization_bits = compute_model_nbits(quantization_compress_model)
print('低秩+量化后', quantization_bits/1000000)
print('压缩比', origin_bits/quantization_bits)
ckpt = 'save_decom/QLTD/4_8_0.001_1024_R18_C100.pt'
torch.save(quantization_compress_model, ckpt)
print(ckpt)

# #

