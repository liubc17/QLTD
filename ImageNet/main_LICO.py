
from calculate_model_size.model_size import compute_model_nbits
import torch
# from lowrank_compress import lowrank_compress_model
from torch import nn
from typing import Callable, Dict, List, NewType, Optional, Set, Union
from config.config_loader import load_config
import os
import torchvision
# from tensorboardX import SummaryWriter
from quantization_compress_utils.quantization_compress import quantization_compress_model
from permutation_LICO import permute_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""load config"""
file_path = os.path.dirname(__file__)
default_config = os.path.join(file_path, "config/train_vgg16_ILSVRC2012.yaml")
config = load_config(file_path, default_config_path=default_config)
model_config = config["model"]
compression_config = model_config["compression_parameters"]

lowrank_compressed_model = torch.load('save_model/VGG16/lowrank/lowrank_2_compressed_model_imagenet_9ep_trained.pt')

if "permutations" in model_config and model_config.get("use_permutations", False):
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
torch.save(quantization_compress_model,
       'save_model/VGG16/LRQ/quantization_2_4_k256_compressed_model_imagenet_df9.pt')

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
