import torch
import torchvision
from torchvision.models.resnet import model_urls
from torchvision.models.alexnet import model_urls
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn

def get_resnet18(pretrained: bool = False) -> torch.nn.Module:
    """Get PyTorch's default ResNet-18 model"""
    # Hack to fix SSL error while loading pretrained model -- see https://github.com/pytorch/pytorch/issues/2271
    model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
    model = torchvision.models.resnet18(pretrained=pretrained)
    model._arch = "resnet18"
    return model


def get_resnet50(pretrained: bool = False) -> torch.nn.Module:
    """Get PyTorch's default ResNet-50 model"""
    model_urls["resnet50"] = model_urls["resnet50"].replace("https://", "http://")
    model = torchvision.models.resnet50(pretrained=pretrained)
    model._arch = "resnet50"
    return model

def get_alexnet(pretrained: bool = False) -> torch.nn.Module:

    model_urls["alexnet"]

def get_resnet50ssl() -> torch.nn.Module:
    """Get a ResNet-50 pre-trained on YFC100M"""
    # Avoid SSL error due to missing python certificates -- see https://stackoverflow.com/a/60671292/1884420
    import ssl

    previous_ssl_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl")
    model._arch = "resnet50"

    ssl._create_default_https_context = previous_ssl_context
    return model


def get_mask_rcnn(pretrained: bool = False) -> torch.nn.Module:
    """Get a mask-RCNN pre-trained on ImageNet and CoCo"""
    torchvision.models.detection.mask_rcnn.model_urls[
        "maskrcnn_resnet50_fpn_coco"
    ] = "http://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    return model


# model = get_resnet50(True)
# print(model)
# torch.save(model,'save_model/origin_model/Resnet50_pretrained_imagenet.pt')
