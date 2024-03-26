import torch
import torch.nn.functional as F 
from Resnet18_torchvision import ResNet18
from Mydataset import get_trainset,get_valset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import model_urls

import torchvision
from calculate_model_size.model_size import compute_model_nbits
MODEL_PATH='Models/model9509.pkl'
TESTSET_LENGTH=50000
device='cuda' if torch.cuda.is_available() else 'cpu'

def get_resnet18(pretrained: bool = False) -> torch.nn.Module:
    """Get PyTorch's default ResNet-18 model"""
    # Hack to fix SSL error while loading pretrained model -- see https://github.com/pytorch/pytorch/issues/2271
    model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
    model = torchvision.models.resnet18(pretrained=pretrained)
    model._arch = "resnet18"
    return model

def test():
    test_loader=get_valset(128)
    original_model = torch.load('save_model/origin_model/Resnet18_pretrained_imagenet.pt')
    print(compute_model_nbits(original_model))
    quantization_model = torch.load('save_model/origin_model/Resnet50_pretrained_imagenet.pt')
    print(quantization_model)
    # quantization_model = get_resnet18(True)
    print('compression_ratio:',compute_model_nbits(original_model) / compute_model_nbits(quantization_model))
    quantization_model.to(device)
    # torch.save(network,'E:/！！！research/Holiday_网络压缩/LRQ/save_model/origin_model/Resnet18_cifar10.pt')
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            preds=quantization_model(images)
            loss=F.cross_entropy(preds,labels)
            a=preds.argmax(dim=1).equal(labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
        test_acc=total_correct/TESTSET_LENGTH
    return test_acc
if __name__=='__main__':
   print(test())