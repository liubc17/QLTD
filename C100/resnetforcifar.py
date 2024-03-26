'''
Deep Residual Learning for Image Recognition
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn

from resnet import ConvBnActBlock, BasicBlock, Bottleneck

__all__ = [
    'resnet18cifar',
    'resnet34halfcifar',
    'resnet34cifar',
    'resnet50halfcifar',
    'resnet50cifar',
    'resnet101cifar',
    'resnet152cifar',
]


class ResNetCifar(nn.Module):
    def __init__(self, block, layer_nums, inplanes=64, num_classes=100):
        super(ResNetCifar, self).__init__()
        self.block = block
        self.layer_nums = layer_nums
        self.num_classes = num_classes
        self.inplanes = inplanes
        self.planes = [inplanes, inplanes * 2, inplanes * 4, inplanes * 8]
        self.expansion = 1 if block is BasicBlock else 4

        self.conv1st = ConvBnActBlock(3,
                                    self.inplanes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    has_bn=True,
                                    has_act=True)

        self.layer1 = self.make_layer(self.block,
                                      self.planes[0],
                                      self.layer_nums[0],
                                      stride=1)
        self.layer2 = self.make_layer(self.block,
                                      self.planes[1],
                                      self.layer_nums[1],
                                      stride=2)
        self.layer3 = self.make_layer(self.block,
                                      self.planes[2],
                                      self.layer_nums[2],
                                      stride=2)
        self.layer4 = self.make_layer(self.block,
                                      self.planes[3],
                                      self.layer_nums[3],
                                      stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.planes[3] * self.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, planes, layer_nums, stride):
        layers = []
        for i in range(0, layer_nums):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))
            self.inplanes = planes * self.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1st(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnetcifar(block, layers, inplanes, **kwargs):
    model = ResNetCifar(block, layers, inplanes, **kwargs)

    return model


def resnet18cifar(**kwargs):
    return _resnetcifar(BasicBlock, [2, 2, 2, 2], 64, **kwargs)


def resnet34halfcifar(**kwargs):
    return _resnetcifar(BasicBlock, [3, 4, 6, 3], 32, **kwargs)


def resnet34cifar(**kwargs):
    return _resnetcifar(BasicBlock, [3, 4, 6, 3], 64, **kwargs)


def resnet50halfcifar(**kwargs):
    return _resnetcifar(Bottleneck, [3, 4, 6, 3], 32, **kwargs)


def resnet50cifar(**kwargs):
    return _resnetcifar(Bottleneck, [3, 4, 6, 3], 64, **kwargs)


def resnet101cifar(**kwargs):
    return _resnetcifar(Bottleneck, [3, 4, 23, 3], 64, **kwargs)


def resnet152cifar(**kwargs):
    return _resnetcifar(Bottleneck, [3, 8, 36, 3], 64, **kwargs)


