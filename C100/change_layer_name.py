import torch
import torchvision
from torchvision.models.resnet import model_urls
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from resnetforcifar import resnet18cifar
x = torch.load('save_model/origin_model/resnet18cifar-acc76.750.pth')
print(x)
#
x['conv1st.layer.0.weight'] = x.pop('conv1.layer.0.weight')
x['conv1st.layer.1.weight'] = x.pop('conv1.layer.1.weight')
x['conv1st.layer.1.bias'] = x.pop('conv1.layer.1.bias')
x['conv1st.layer.1.running_mean'] = x.pop('conv1.layer.1.running_mean')
x['conv1st.layer.1.running_var'] = x.pop('conv1.layer.1.running_var')
x['conv1st.layer.1.num_batches_tracked'] = x.pop('conv1.layer.1.num_batches_tracked')
#
# #
torch.save(x,'save_model/origin_model/changename_utils/R18_cifar100_rename_dict.pth')

space_model = resnet18cifar()
change_dict = torch.load('save_model/origin_model/changename_utils/R18_cifar100_rename_dict.pth')
space_model.load_state_dict(change_dict)
print(space_model)

torch.save(space_model,'save_model/origin_model/Resnet18_cifar100.pt')