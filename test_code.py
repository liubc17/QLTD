import torch
from torchvision.models import resnet50,resnet18
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from calculate_model_size.model_size import compute_model_nbits
from ptflops import get_model_complexity_info
# model = resnet18(num_classes = 1000)
low_rank_model = torch.load("save_model/lowrank/lowrank_2_compressed_model_cifar10.pt")

flops,params = get_model_complexity_info(low_rank_model, (3,32,32),as_strings=True,print_per_layer_stat=True)
print("*****************************************")
original_model = torch.load("save_model/origin_model/Resnet18_cifar10.pt")
flops2,params2 = get_model_complexity_info(original_model, (3,32,32),as_strings=True,print_per_layer_stat=True)

# print('total bits:',compute_model_nbits(model))
# layer1 = model.layer1
# total = 0
# for name,param in model.named_parameters():
#     print(name)
#     print(param.nelement())
#     total += param.nelement()
#
# print(total)
# tensor = (torch.rand(1, 3, 32, 32).cuda(),)
# #
# # # 分析FLOPs
# flops2 = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops2.total() / 1e6)