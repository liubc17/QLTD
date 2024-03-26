import torch
import torchvision
from torchvision.models.resnet import model_urls
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
from get_pretrained_imagenet_model import get_resnet18, get_resnet50, get_resnet50ssl, get_mask_rcnn
from Resnet18_torchvision import ResNet18
from Resnet50_torchvision import ResNet50
x = torch.load('save_model/origin_model/Resnet50_pretrained_imagenet.pth')
# print(x)
# older_val = dict['conv1.weight']
older_val = x['conv1.weight']
x['conv1st.weight'] = x.pop('conv1.weight')

torch.save(x, 'save_model/origin_model/changename_utils/Resnet50_pretrained_imagenet_rename_dict.pt')

space_model = ResNet50()
changed_dict = torch.load('save_model/origin_model/Resnet50_pretrained_imagenet_rename_dict.pt')
space_model.load_state_dict(changed_dict)
print(space_model)
torch.save(space_model,'save_model/origin_model/Resnet50_pretrained_imagenet.pt')
# #
# changed_dict = torch.load('save_model/orgin_model/Resnet18_pretrained_imagenet_rename.pt')
# print(changed_dict)
#
# space_model = ResNet18()
# changed_dict = torch.load('save_model/orgin_model/Resnet18_pretrained_imagenet_rename.pt')
# # print(changed_dict)
# space_model.load_state_dict(changed_dict)
# torch.save(space_model,'save_model/orgin_model/Resnet18_pretrained_imagenet.pt')
# print(space_model)


